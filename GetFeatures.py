import os
import yaml
import requests
import numpy as np
import pandas as pd
import osmnx as ox
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Point


class GetFeatures:
    """
    Reads a route_vars CSV and extracts per-segment features:
    road type, slope, weather, and aggregated EV variables.
    """

    def __init__(self, config_path='config.yml', trip_date=None, input_file=None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        processed = self.config['processed_data_path']
        if trip_date:
            processed = os.path.join(processed, trip_date)
        self.processed_path = processed

        route_file = input_file or self.config['output_files']['route_vars']
        self.route_vars_path = os.path.join(processed, route_file)

        feat_cfg = self.config.get('features', {})
        self.segment_length_m = feat_cfg.get('segment_length_meters', 100)
        self.weather_api_key = feat_cfg.get('wwo_api_key')
        self.output_file = feat_cfg.get('output_file', 'features.csv')
        self.output_path = os.path.join(processed, self.output_file)

        self.df = None
        self.segments_df = None

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        la1, lo1 = radians(lat1), radians(lon1)
        la2, lo2 = radians(lat2), radians(lon2)
        dlat, dlon = la2 - la1, lo2 - lo1
        a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    # ── 1. load ────────────────────────────────────────────────────────
    def load(self):
        self.df = pd.read_csv(self.route_vars_path)
        required = {'latitude', 'longitude'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df['Hour'] = self.df['datetime'].dt.hour
        else:
            from datetime import datetime, timezone
            self.df['Hour'] = datetime.now(timezone.utc).hour
        print(f"Loaded {len(self.df)} points from {self.route_vars_path}")
        return self.df

    # ── 2. distances ───────────────────────────────────────────────────
    def compute_distances(self):
        deltas = [0.0]
        for i in range(1, len(self.df)):
            d = self._haversine_m(
                self.df.loc[i - 1, 'latitude'], self.df.loc[i - 1, 'longitude'],
                self.df.loc[i, 'latitude'], self.df.loc[i, 'longitude'])
            deltas.append(d)
        self.df['Delta_d_raw'] = deltas
        self.df['Trip_distance'] = np.cumsum(deltas)
        print(f"  Total distance: {self.df['Trip_distance'].iloc[-1] / 1000:.2f} km")

    # ── 3. segments ────────────────────────────────────────────────────
    def assign_segments(self):
        seg_ids, seg, acc = [], 0, 0.0
        for d in self.df['Delta_d_raw']:
            if acc >= self.segment_length_m:
                seg += 1
                acc = 0.0
            seg_ids.append(seg)
            acc += d
        self.df['segment_id'] = seg_ids

        seg_lengths = (self.df.groupby('segment_id')['Delta_d_raw']
                       .sum().reset_index(name='Delta_d'))
        self.df = self.df.merge(seg_lengths, on='segment_id', how='left')
        print(f"  Segments: {self.df['segment_id'].nunique()} (target {self.segment_length_m}m)")

    # ── 4. elevation & slope ───────────────────────────────────────────
    def fetch_elevation(self):
        self.df['Elevation'] = np.nan
        for seg_id, grp in self.df.groupby('segment_id'):
            if len(grp) < 2:
                continue
            p0, p1 = grp.iloc[0], grp.iloc[-1]
            loc = f"{p0['latitude']},{p0['longitude']}|{p1['latitude']},{p1['longitude']}"
            try:
                r = requests.get(
                    "https://api.open-elevation.com/api/v1/lookup",
                    params={"locations": loc}, timeout=10)
                r.raise_for_status()
                res = r.json()["results"]
                self.df.loc[grp.index[0], 'Elevation_start'] = res[0]['elevation']
                self.df.loc[grp.index[-1], 'Elevation_end'] = res[1]['elevation']
            except Exception:
                pass
        print("  Elevation fetched")

    def compute_slope(self):
        slope_map = {}
        for seg_id, grp in self.df.groupby('segment_id'):
            z0 = grp['Elevation_start'].iloc[0] if 'Elevation_start' in grp else np.nan
            z1 = grp['Elevation_end'].iloc[-1] if 'Elevation_end' in grp else np.nan
            d_seg = grp['Delta_d'].iloc[0]
            if pd.isna(z0) or pd.isna(z1) or d_seg <= 0:
                slope_map[seg_id] = 0.0
            else:
                slope_map[seg_id] = np.clip(
                    np.degrees(np.arctan((z1 - z0) / d_seg)), -15, 15)
        self.df['Slope'] = self.df['segment_id'].map(slope_map)
        print("  Slope computed")

    # ── 5. OSM road features ──────────────────────────────────────────
    def fetch_road_features(self):
        for col in ['primary', 'residential', 'secondary',
                     'crossing', 'tertiary', 'give_way']:
            self.df[col] = 0

        try:
            lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
            lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()
            bbox = (lat_max + 0.005, lat_min - 0.005,
                    lon_max + 0.005, lon_min - 0.005)
            G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
            nodes, edges = ox.graph_to_gdfs(G)
            utm_crs = ox.projection.project_gdf(nodes).crs
            nodes_p = nodes.to_crs(utm_crs)
            edges_p = edges.to_crs(utm_crs)

            for i, row in self.df.iterrows():
                p = Point(row['longitude'], row['latitude'])
                p_proj = ox.projection.project_geometry(p, to_crs=utm_crs)[0]

                # Edge (road type)
                edge_idx = edges_p.distance(p_proj).idxmin()
                hw = edges.loc[edge_idx].get('highway', '')
                if isinstance(hw, list):
                    hw = hw[0]
                if hw in ('primary', 'residential', 'secondary'):
                    self.df.loc[i, hw] = 1

                # Node (intersection type)
                node_idx = nodes_p.distance(p_proj).idxmin()
                nhw = nodes.loc[node_idx].get('highway', '')
                if nhw in ('crossing', 'tertiary', 'give_way'):
                    self.df.loc[i, nhw] = 1

            print("  Road features extracted")
        except Exception as e:
            print(f"  Road features error: {e}")

    # ── 6. weather (WWO) ──────────────────────────────────────────────
    def fetch_weather(self):
        self.df['Hum'] = np.nan
        self.df['OAT[DegC]_API'] = np.nan
        self.df['precipMM'] = np.nan

        if not self.weather_api_key:
            print("  No WWO API key configured – skipping weather")
            return

        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        if 'datetime' not in self.df.columns:
            self.df['_weather_dt'] = now_utc

        try:
            chunk_km = 20
            max_km = self.df['Trip_distance'].max() / 1000
            n_chunks = max(1, int(np.ceil(max_km / chunk_km)))
            weather_data = []

            for ci in range(n_chunks):
                start_m = ci * chunk_km * 1000
                end_m = min((ci + 1) * chunk_km * 1000, max_km * 1000)
                mid_m = (start_m + end_m) / 2
                idx = (self.df['Trip_distance'] - mid_m).abs().idxmin()
                lat = self.df.loc[idx, 'latitude']
                lon = self.df.loc[idx, 'longitude']
                dt_col = 'datetime' if 'datetime' in self.df.columns else '_weather_dt'
                dt = pd.Timestamp(self.df.loc[idx, dt_col])

                try:
                    r = requests.get(
                        "https://api.worldweatheronline.com/premium/v1/past-weather.ashx",
                        params={
                            "key": self.weather_api_key,
                            "q": f"{lat},{lon}",
                            "format": "json",
                            "date": dt.strftime("%Y-%m-%d"),
                            "tp": 1,
                        }, timeout=10)
                    r.raise_for_status()
                    hourly = r.json()["data"]["weather"][0]["hourly"]
                    closest = min(hourly,
                                  key=lambda x: abs(int(x["time"]) // 100 - dt.hour))
                    weather_data.append({
                        "start_m": start_m, "end_m": end_m,
                        "Hum": float(closest["humidity"]),
                        "OAT": float(closest["tempC"]),
                        "precip": float(closest["precipMM"]),
                    })
                except Exception:
                    prev = weather_data[-1] if weather_data else {
                        "Hum": 65.0, "OAT": 25.0, "precip": 0.0}
                    weather_data.append({
                        "start_m": start_m, "end_m": end_m,
                        "Hum": prev["Hum"], "OAT": prev["OAT"],
                        "precip": prev["precip"],
                    })

            for i, row in self.df.iterrows():
                td = row['Trip_distance']
                for wd in weather_data:
                    if wd['start_m'] <= td <= wd['end_m']:
                        self.df.loc[i, 'Hum'] = wd['Hum']
                        self.df.loc[i, 'OAT[DegC]_API'] = wd['OAT']
                        self.df.loc[i, 'precipMM'] = wd['precip']
                        break
            print(f"  Weather fetched ({n_chunks} location(s))")
        except Exception as e:
            print(f"  Weather error: {e}")
            self.df['Hum'] = 65.0
            self.df['OAT[DegC]_API'] = 25.0
            self.df['precipMM'] = 0.0

        self.df.drop(columns='_weather_dt', errors='ignore', inplace=True)

    # ── 7. aggregate ──────────────────────────────────────────────────
    def aggregate(self):
        # Always-present columns after processing
        agg = {
            'Trip_distance': 'last',
            'Delta_d': 'first',
        }

        # Optional columns: include only if they exist in the dataframe
        optional_last = ['Hour']
        optional_mean = ['Hum', 'OAT[DegC]_API', 'precipMM', 'Slope',
                         'Voltage', 'Current', 'Power', 'SoC']
        optional_max = ['primary', 'residential', 'secondary',
                        'crossing', 'tertiary', 'give_way']

        for col in optional_last:
            if col in self.df.columns:
                agg[col] = 'last'
        for col in optional_mean:
            if col in self.df.columns:
                agg[col] = 'mean'
        for col in optional_max:
            if col in self.df.columns:
                agg[col] = 'max'

        result = self.df.groupby('segment_id', as_index=False).agg(agg)
        result = result[result['Delta_d'] > 0].reset_index(drop=True)
        result['segment_id'] = range(len(result))

        col_order = ['segment_id'] + [c for c in
                     ['Hour', 'Trip_distance', 'Delta_d',
                      'Hum', 'OAT[DegC]_API', 'precipMM', 'Slope',
                      'primary', 'residential', 'secondary',
                      'crossing', 'tertiary', 'give_way',
                      'Voltage', 'Current', 'Power', 'SoC']
                     if c in result.columns]
        self.segments_df = result[col_order]
        print(f"  Aggregated into {len(self.segments_df)} segments")
        return self.segments_df

    # ── 8. save ───────────────────────────────────────────────────────
    def save(self):
        if self.segments_df is not None:
            self.segments_df.to_csv(self.output_path, index=False)
            print(f"Saved features → {self.output_path}")

    # ── run all ───────────────────────────────────────────────────────
    def run(self):
        self.load()
        self.compute_distances()
        self.assign_segments()
        self.fetch_weather()
        self.fetch_road_features()
        self.fetch_elevation()
        self.compute_slope()
        result = self.aggregate()
        self.save()
        return result
