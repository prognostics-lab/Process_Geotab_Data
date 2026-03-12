import os
import yaml
import requests
import numpy as np
import pandas as pd
import folium
from math import radians, sin, cos, sqrt, atan2


class SelectRoutes:
    """
    Given an origin, destination, and optional waypoints, compute a driving
    route and return GPS coordinates sampled every *interval_meters*.
    Uses the OSRM public API for routing, travel time, and distance.
    """

    def __init__(self, config_path='config.yml', trip_date=None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        rt = self.config.get('route', {})
        self.interval_meters = rt.get('interval_meters', 10)
        self.output_file = rt.get('output_file', 'route_coords.csv')
        self.processed_path = self.config['processed_data_path']
        if trip_date:
            self.processed_path = os.path.join(self.processed_path, trip_date)
        self.output_path = os.path.join(self.processed_path, self.output_file)

        self.route_geometry = None
        self.gps_coords = None
        self.travel_info = None

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        la1, lo1 = radians(lat1), radians(lon1)
        la2, lo2 = radians(lat2), radians(lon2)
        dlat, dlon = la2 - la1, lo2 - lo1
        a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    @staticmethod
    def _decode_polyline(encoded):
        """Decode a Google-encoded polyline string into (lat, lon) tuples."""
        coords = []
        i, lat, lng = 0, 0, 0
        while i < len(encoded):
            for val_ref in ('lat', 'lng'):
                shift, result = 0, 0
                while True:
                    b = ord(encoded[i]) - 63
                    i += 1
                    result |= (b & 0x1F) << shift
                    shift += 5
                    if b < 0x20:
                        break
                delta = (~(result >> 1)) if (result & 1) else (result >> 1)
                if val_ref == 'lat':
                    lat += delta
                else:
                    lng += delta
            coords.append((lat / 1e5, lng / 1e5))
        return coords

    # ── 1. find route via OSRM ────────────────────────────────────────
    def find_route(self, origin, destination, waypoints=None, profile='driving'):
        """
        Find a driving route via the OSRM API, passing through waypoints.

        Parameters
        ----------
        origin : tuple (lat, lon)
        destination : tuple (lat, lon)
        waypoints : list of (lat, lon) tuples, optional
        profile : str  ('driving', 'walking', or 'cycling')
        """
        self.origin = origin
        self.destination = destination
        self.waypoints = waypoints or []

        valid_profiles = {'driving', 'walking', 'cycling'}
        if profile not in valid_profiles:
            raise ValueError(f"Invalid profile '{profile}'. Options: {valid_profiles}")

        # Build coordinate string: origin ; wp1 ; wp2 ; ... ; destination
        all_pts = [origin] + self.waypoints + [destination]
        osrm_coords = ';'.join(f"{lon},{lat}" for lat, lon in all_pts)

        url = f"http://router.project-osrm.org/route/v1/{profile}/{osrm_coords}"
        params = {'overview': 'full', 'geometries': 'polyline', 'steps': 'false'}

        print("Requesting route from OSRM...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if data['code'] != 'Ok':
            raise ValueError(f"OSRM error: {data.get('message', data['code'])}")

        route = data['routes'][0]
        self.route_geometry = self._decode_polyline(route['geometry'])

        duration_s = route['duration']
        distance_m = route['distance']

        self.travel_info = {
            'duration_seconds': round(duration_s, 1),
            'duration_minutes': round(duration_s / 60, 2),
            'distance_km': round(distance_m / 1000, 2),
            'origin': self.origin,
            'destination': self.destination,
        }

        print(f"Route found: {len(self.route_geometry)} geometry points, "
              f"{self.travel_info['distance_km']} km, "
              f"{self.travel_info['duration_minutes']} min")
        return self.route_geometry

    # ── 2. interpolate GPS points along route ─────────────────────────
    def interpolate_route(self, interval_meters=None):
        """
        Resample the OSRM route geometry every *interval_meters*.
        """
        if self.route_geometry is None:
            raise ValueError("Call find_route() first.")

        interval = interval_meters or self.interval_meters
        geom = self.route_geometry
        points = [geom[0]]
        residual = 0.0

        for k in range(len(geom) - 1):
            lat1, lon1 = geom[k]
            lat2, lon2 = geom[k + 1]
            seg_len = self._haversine_m(lat1, lon1, lat2, lon2)
            if seg_len == 0:
                continue

            dist_along = residual
            while dist_along < seg_len:
                frac = dist_along / seg_len
                points.append((
                    lat1 + (lat2 - lat1) * frac,
                    lon1 + (lon2 - lon1) * frac,
                ))
                dist_along += interval
            residual = dist_along - seg_len

        # Ensure the final point is included
        if points[-1] != geom[-1]:
            points.append(geom[-1])

        self.gps_coords = pd.DataFrame(points, columns=['latitude', 'longitude'])
        print(f"Interpolated {len(self.gps_coords)} GPS points (every {interval} m)")
        return self.gps_coords

    # ── 4. save map ───────────────────────────────────────────────────
    def save_map(self):
        """Create an interactive Folium HTML map of the route."""
        if self.gps_coords is None:
            raise ValueError("Call interpolate_route() first.")

        coords = list(zip(self.gps_coords['latitude'], self.gps_coords['longitude']))
        centre = [np.mean(self.gps_coords['latitude']),
                  np.mean(self.gps_coords['longitude'])]

        m = folium.Map(location=centre, zoom_start=12)

        # Route polyline
        folium.PolyLine(coords, color='blue', weight=4, opacity=0.8).add_to(m)

        # Origin marker
        folium.Marker(
            location=coords[0],
            popup='Origin',
            icon=folium.Icon(color='green', icon='play'),
        ).add_to(m)

        # Destination marker
        folium.Marker(
            location=coords[-1],
            popup='Destination',
            icon=folium.Icon(color='red', icon='stop'),
        ).add_to(m)

        # Waypoint markers
        for i, wp in enumerate(self.waypoints):
            folium.Marker(
                location=wp,
                popup=f'Waypoint {i + 1}',
                icon=folium.Icon(color='orange', icon='flag'),
            ).add_to(m)

        os.makedirs(self.processed_path, exist_ok=True)
        map_path = os.path.join(self.processed_path, 'route_map.html')
        m.save(map_path)
        print(f"Saved map  → {map_path}")
        return m

    # ── 5. save ───────────────────────────────────────────────────────
    def save(self):
        if self.gps_coords is not None:
            os.makedirs(self.processed_path, exist_ok=True)
            self.gps_coords.to_csv(self.output_path, index=False)
            print(f"Saved route → {self.output_path} ({len(self.gps_coords)} points)")
            self.save_map()

    # ── run all ───────────────────────────────────────────────────────
    def run(self, origin, destination, waypoints=None, profile='driving'):
        """
        Full pipeline: find route → interpolate → save.
        """
        self.find_route(origin, destination, waypoints, profile)
        self.interpolate_route()
        self.save()
        return self.gps_coords, self.travel_info
