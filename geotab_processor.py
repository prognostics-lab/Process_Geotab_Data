import os
import yaml
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

class GeotabProcessor:
    def __init__(self, config_path='config.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.raw_path = self.config['raw_data_path']
        self.processed_path = self.config['processed_data_path']
        tw = self.config.get('time_window', {})
        self.time_start = pd.to_datetime(tw['start']) if tw.get('start') else None
        self.time_end = pd.to_datetime(tw['end']) if tw.get('end') else None
        self.vars_df = None
        self.gps_df = None
        self.route_vars_df = None

        # Extract trip date from raw data metadata
        self.trip_date = self._extract_trip_date()
        self.processed_path = os.path.join(self.config['processed_data_path'], self.trip_date)
        print(f"Trip date: {self.trip_date}")

    def _extract_trip_date(self):
        """Read the FromDate from the raw motor status file metadata."""
        f1 = os.path.join(self.raw_path, self.config['files']['motor_status'])
        meta = pd.read_excel(f1, header=None, nrows=5, usecols=[0, 1], engine='openpyxl')
        for i, row in meta.iterrows():
            if str(row.iloc[0]).strip() == 'FromDate':
                dt = pd.to_datetime(row.iloc[1])
                return dt.strftime('%Y-%m-%d')
        return 'unknown_date'

    def _get_input_path(self, key):
        return os.path.join(self.raw_path, self.config['files'][key])

    def _get_output_path(self, key):
        return os.path.join(self.processed_path, self.config['output_files'][key])

    def process_vars(self):
        """Process motor status report into vars DataFrame (datetime, Voltage, Current, Power, SoC, Procedencia)."""
        df = pd.read_excel(self._get_input_path('motor_status'), header=8, engine='openpyxl')

        # Extract datetime, variable name, and value
        raw = df[[df.columns[5], df.columns[6], df.columns[13]]].copy()
        raw.columns = ['datetime', 'variable', 'valor']
        raw['datetime'] = pd.to_datetime(raw['datetime'])
        raw['valor'] = pd.to_numeric(raw['valor'], errors='coerce')
        raw = raw.dropna(subset=['datetime', 'variable', 'valor'])

        # Apply time window filter
        if self.time_start:
            raw = raw[raw['datetime'] >= self.time_start]
        if self.time_end:
            raw = raw[raw['datetime'] <= self.time_end]

        # Split by variable type
        voltage = raw[raw['variable'].str.contains('Voltaje', case=False)][['datetime', 'valor']].rename(columns={'valor': 'Voltage'}).sort_values('datetime')
        power = raw[raw['variable'].str.contains('Energía', case=False)][['datetime', 'valor']].rename(columns={'valor': 'Power'}).sort_values('datetime')
        soc = raw[raw['variable'].str.contains('Estado de carga', case=False)][['datetime', 'valor']].rename(columns={'valor': 'SoC'}).sort_values('datetime')

        # Merge Voltage and SoC into Power timestamps by nearest match (30s tolerance)
        result = pd.merge_asof(power, voltage, on='datetime', direction='nearest', tolerance=pd.Timedelta('30s'))
        result = pd.merge_asof(result, soc, on='datetime', direction='nearest', tolerance=pd.Timedelta('30s'))

        # Linear interpolation to fill gaps beyond tolerance
        result['Voltage'] = result['Voltage'].interpolate(method='linear', limit_direction='both')
        result['SoC'] = result['SoC'].interpolate(method='linear', limit_direction='both')

        # Calculate Current = Power / Voltage
        result['Current'] = result['Power'] / result['Voltage']

        # Calculate cumulative Energy in Wh (trapezoidal integration of Power over time)
        dt_s = result['datetime'].diff().dt.total_seconds().fillna(0)
        result['Energy'] = (result['Power'] * dt_s / 3600).cumsum()

        # Add provenance column
        result['Procedencia'] = 'AA'

        # Reorder columns
        self.vars_df = result[['datetime', 'Voltage', 'Current', 'Power', 'Energy', 'SoC', 'Procedencia']].reset_index(drop=True)

        print(f"vars: {self.vars_df.shape[0]} rows")
        return self.vars_df

    def process_gps(self):
        """Process data log into GPS DataFrame (datetime, latitude, longitude)."""
        df = pd.read_excel(self._get_input_path('data_log'), header=9, engine='openpyxl')

        self.gps_df = df[df['DebugRecordType'] == 'GpsRecord'][['DebugDateTime', 'DebugLatitude', 'DebugLongitude']].copy()
        self.gps_df.columns = ['datetime', 'latitude', 'longitude']
        self.gps_df['datetime'] = pd.to_datetime(self.gps_df['datetime'])
        self.gps_df['latitude'] = pd.to_numeric(self.gps_df['latitude'], errors='coerce')
        self.gps_df['longitude'] = pd.to_numeric(self.gps_df['longitude'], errors='coerce')
        self.gps_df = self.gps_df.dropna()

        # Apply time window filter
        if self.time_start:
            self.gps_df = self.gps_df[self.gps_df['datetime'] >= self.time_start]
        if self.time_end:
            self.gps_df = self.gps_df[self.gps_df['datetime'] <= self.time_end]

        self.gps_df = self.gps_df.reset_index(drop=True)

        print(f"gps: {self.gps_df.shape[0]} rows")
        return self.gps_df

    def merge_route_vars(self):
        """Merge vars into GPS by nearest timestamp with interpolation."""
        if self.gps_df is None or self.vars_df is None:
            raise ValueError("Run process_gps() and process_vars() first.")

        gps = self.gps_df[['datetime', 'latitude', 'longitude']].copy()
        features = self.vars_df.copy()

        # Fix date parsing if day/month are swapped
        if gps['datetime'].dt.month.iloc[0] != features['datetime'].dt.month.iloc[0]:
            gps['datetime'] = gps['datetime'].apply(
                lambda x: x.replace(month=x.day, day=x.month) if x.day <= 12 else x
            )

        gps = gps.sort_values('datetime').reset_index(drop=True)
        features = features.sort_values('datetime').reset_index(drop=True)

        # Merge by nearest timestamp (tolerance 30s)
        merged = pd.merge_asof(gps, features, on='datetime', direction='nearest',
                               tolerance=pd.Timedelta('30s'))

        # Linear interpolation for numeric columns
        for col in ['Voltage', 'Current', 'Power', 'Energy', 'SoC']:
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')

        # Forward/backward fill for categorical column
        merged['Procedencia'] = merged['Procedencia'].ffill().bfill()

        self.route_vars_df = merged
        print(f"route_vars: {self.route_vars_df.shape[0]} rows")
        return self.route_vars_df

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        """Haversine distance in meters between two points."""
        R = 6371000.0
        la1, lo1, la2, lo2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
        dlat, dlon = la2 - la1, lo2 - lo1
        a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    def densify_gps(self):
        """Interpolate intermediate GPS points so no two consecutive points exceed max gap."""
        if self.route_vars_df is None:
            raise ValueError("Run merge_route_vars() first.")

        max_gap = self.config.get('densify_max_gap_meters', 100)
        df = self.route_vars_df.copy()
        numeric_cols = ['Voltage', 'Current', 'Power', 'Energy', 'SoC']
        original_len = len(df)
        rows = []

        for i in range(len(df)):
            if i == 0:
                rows.append(df.iloc[i].to_dict())
                continue

            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            dist = self._haversine_m(prev['latitude'], prev['longitude'],
                                     curr['latitude'], curr['longitude'])

            if dist > max_gap:
                n_segments = int(np.ceil(dist / max_gap))
                for j in range(1, n_segments):
                    frac = j / n_segments
                    new_row = curr.to_dict()
                    new_row['latitude'] = prev['latitude'] + (curr['latitude'] - prev['latitude']) * frac
                    new_row['longitude'] = prev['longitude'] + (curr['longitude'] - prev['longitude']) * frac
                    # Interpolate datetime
                    dt_prev = prev['datetime'].value
                    dt_curr = curr['datetime'].value
                    new_row['datetime'] = pd.Timestamp(dt_prev + (dt_curr - dt_prev) * frac)
                    # Interpolate numeric columns
                    for col in numeric_cols:
                        v_prev = prev[col]
                        v_curr = curr[col]
                        if pd.notna(v_prev) and pd.notna(v_curr):
                            new_row[col] = v_prev + (v_curr - v_prev) * frac
                    rows.append(new_row)

            rows.append(curr.to_dict())

        self.route_vars_df = pd.DataFrame(rows).reset_index(drop=True)
        print(f"densify: {original_len} → {len(self.route_vars_df)} points (max gap {max_gap}m)")
        return self.route_vars_df

    def save(self):
        """Save vars, gps, and route_vars DataFrames to CSV files."""
        os.makedirs(self.processed_path, exist_ok=True)

        if self.vars_df is not None:
            path = self._get_output_path('vars')
            self.vars_df.to_csv(path, index=False)
            print(f"Saved vars → {path}  ({self.vars_df.shape[0]} rows)")

        if self.gps_df is not None:
            path = self._get_output_path('gps')
            self.gps_df.to_csv(path, index=False)
            print(f"Saved gps  → {path}  ({self.gps_df.shape[0]} rows)")

        if self.route_vars_df is not None:
            path = self._get_output_path('route_vars')
            self.route_vars_df.to_csv(path, index=False)
            print(f"Saved route_vars → {path}  ({self.route_vars_df.shape[0]} rows)")

    def run(self):
        """Process all files and save outputs."""
        self.process_vars()
        self.process_gps()
        self.merge_route_vars()
        self.densify_gps()
        self.save()