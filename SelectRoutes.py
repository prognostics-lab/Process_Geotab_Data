import os
import yaml
import requests
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2


class SelectRoutes:
    """
    Given an origin, destination, and optional waypoints, compute a driving
    route and return GPS coordinates sampled every *interval_meters*.
    Also estimates travel time and distance via the OSRM API.
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

        self.G = None
        self.route_nodes = None
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

    # ── 1. build graph & find route ───────────────────────────────────
    def find_route(self, origin, destination, waypoints=None):
        """
        Find a driving route from origin to destination through optional waypoints.

        Parameters
        ----------
        origin : tuple (lat, lon)
        destination : tuple (lat, lon)
        waypoints : list of (lat, lon) tuples, optional
        """
        self.origin = origin
        self.destination = destination
        self.waypoints = waypoints or []

        # Bounding box from all points
        all_pts = [origin, destination] + self.waypoints
        lats = [p[0] for p in all_pts]
        lons = [p[1] for p in all_pts]
        bbox = (max(lats) + 0.05, min(lats) - 0.05,
                max(lons) + 0.05, min(lons) - 0.05)

        print("Downloading road network...")
        self.G = ox.graph_from_bbox(bbox=bbox, network_type='drive')

        # Snap all points to nearest graph nodes
        orig_node = ox.distance.nearest_nodes(self.G, X=origin[1], Y=origin[0])
        dest_node = ox.distance.nearest_nodes(self.G, X=destination[1], Y=destination[0])

        wp_nodes = []
        for wp in self.waypoints:
            wp_nodes.append(ox.distance.nearest_nodes(self.G, X=wp[1], Y=wp[0]))

        # Build segment list: origin → wp1 → wp2 → ... → destination
        segments = [orig_node] + wp_nodes + [dest_node]

        full_route = []
        total_length = 0.0
        for j in range(len(segments) - 1):
            seg_route = nx.shortest_path(self.G, segments[j], segments[j + 1], weight='length')
            seg_len = sum(self.G[u][v][0]['length']
                          for u, v in zip(seg_route[:-1], seg_route[1:]))
            total_length += seg_len
            if j == 0:
                full_route.extend(seg_route)
            else:
                full_route.extend(seg_route[1:])

        self.route_nodes = full_route
        print(f"Route found: {len(full_route)} nodes, {total_length / 1000:.2f} km")
        return full_route

    # ── 2. interpolate GPS points along edges ─────────────────────────
    def interpolate_route(self, interval_meters=None):
        """
        Generate GPS coordinates along the route every *interval_meters*.
        """
        if self.route_nodes is None:
            raise ValueError("Call find_route() first.")

        interval = interval_meters or self.interval_meters
        G = self.G
        points = []

        for u, v in zip(self.route_nodes[:-1], self.route_nodes[1:]):
            lat1 = G.nodes[u]['y']
            lon1 = G.nodes[u]['x']
            lat2 = G.nodes[v]['y']
            lon2 = G.nodes[v]['x']
            edge_len = G[u][v][0]['length']
            n_pts = max(2, int(edge_len / interval))

            for k in range(n_pts):
                frac = k / n_pts
                points.append((
                    lat1 + (lat2 - lat1) * frac,
                    lon1 + (lon2 - lon1) * frac,
                ))

        # Add final point
        last = self.route_nodes[-1]
        points.append((G.nodes[last]['y'], G.nodes[last]['x']))

        self.gps_coords = pd.DataFrame(points, columns=['latitude', 'longitude'])
        print(f"Interpolated {len(self.gps_coords)} GPS points (every {interval}m)")
        return self.gps_coords

    # ── 3. estimate travel time via OSRM ──────────────────────────────
    def estimate_travel_time(self, profile='driving'):
        """
        Estimate travel time and distance using the OSRM public API.

        Parameters
        ----------
        profile : str
            'driving', 'walking', or 'cycling'.
        """
        if self.gps_coords is None:
            raise ValueError("Call interpolate_route() first.")

        valid_profiles = {'driving', 'walking', 'cycling'}
        if profile not in valid_profiles:
            raise ValueError(f"Invalid profile '{profile}'. Options: {valid_profiles}")

        coords_list = list(zip(self.gps_coords['latitude'],
                               self.gps_coords['longitude']))
        # Ensure origin and destination are exact
        coords_list[0] = self.origin
        coords_list[-1] = self.destination

        osrm_coords = [f"{lon},{lat}" for lat, lon in coords_list]

        # OSRM accepts max ~100 coordinates
        max_coords = 100
        if len(osrm_coords) > max_coords:
            indices = np.linspace(0, len(osrm_coords) - 1, max_coords, dtype=int)
            indices[0] = 0
            indices[-1] = len(osrm_coords) - 1
            osrm_coords = [osrm_coords[i] for i in indices]

        coords_str = ';'.join(osrm_coords)
        url = f"http://router.project-osrm.org/route/v1/{profile}/{coords_str}"
        params = {'overview': 'false', 'steps': 'false'}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if data['code'] != 'Ok':
            raise ValueError(f"OSRM error: {data.get('message', data['code'])}")

        route = data['routes'][0]
        duration_s = route['duration']
        distance_m = route['distance']

        self.travel_info = {
            'duration_seconds': round(duration_s, 1),
            'duration_minutes': round(duration_s / 60, 2),
            'distance_km': round(distance_m / 1000, 2),
            'origin': self.origin,
            'destination': self.destination,
        }

        print(f"Travel estimate: {self.travel_info['distance_km']} km, "
              f"{self.travel_info['duration_minutes']} min")
        return self.travel_info

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
        Full pipeline: find route → interpolate → estimate time → save.
        """
        self.find_route(origin, destination, waypoints)
        self.interpolate_route()
        self.estimate_travel_time(profile)
        self.save()
        return self.gps_coords, self.travel_info
