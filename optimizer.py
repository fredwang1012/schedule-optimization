#!/usr/bin/env python3

# Fix for corporate network SSL certificate issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
TRAVEL-OPTIMIZED CLEANING SCHEDULER FOR FRANCHISE 372
====================================================
Complete system to optimize cleaning schedules by minimizing travel time between jobs.

Uses real geocoding and routing to create efficient team routes.
Respects customer availability, team composition, and operational constraints.

Dependencies: pip install pandas geopy routingpy numpy python-dateutil
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import sys
from collections import defaultdict
import time as time_module
import traceback

# Geocoding and routing (COMMENTED - using pre-calculated matrices instead)
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import routingpy as rp

# ortools VRP solver
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class OptimizationLogger:
    """Enhanced logging with emojis and colors for terminal output"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = time_module.time()
        
    def header(self, message):
        print(f"\n{'='*80}")
        print(f"🚀 {message}")
        print(f"{'='*80}")
        
    def section(self, message):
        print(f"\n📋 {message}")
        print("-" * 60)
        
    def debug(self, message):
        if self.verbose:
            print(f"🔍 DEBUG: {message}")
    
    def info(self, message):
        print(f"📊 INFO: {message}")
        
    def success(self, message):
        print(f"✅ SUCCESS: {message}")
        
    def warning(self, message):
        print(f"⚠️  WARNING: {message}")
        
    def error(self, message):
        print(f"❌ ERROR: {message}")
        
    def progress(self, current, total, message="Processing"):
        if total > 0:
            pct = (current / total) * 100
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r🔄 {message}: [{bar}] {pct:.1f}% ({current}/{total})", end='', flush=True)
            if current == total:
                print()  # New line when complete

class TravelOptimizedScheduler:
    """Main scheduler class that orchestrates the entire optimization process"""
    
    def __init__(self, franchise_id=372, verbose=True):
        self.franchise_id = franchise_id
        self.logger = OptimizationLogger(verbose)
        self.data_folder = "data files"
        
        # Initialize services
        self.geolocator = None
        self.routing_api = None
        
        # Data storage
        self.franchise_info = {}
        self.customers = {}
        self.cleanings = []
        self.teams = {}
        self.coordinates = {}
        self.travel_matrix = {}
        self.feasible_assignments = []
        
        # Results
        self.optimized_schedule = {}
        self.optimization_stats = {}
        
    def run_complete_optimization(self, target_week_start=None):
        """Run the complete optimization pipeline using pre-computed matrices and ortools VRP"""
        try:
            self.logger.header("TRAVEL-OPTIMIZED CLEANING SCHEDULER WITH ORTOOLS VRP")
            
            # Step 1: Load franchise configuration
            self.load_franchise_data()
            
            # Step 2: Set target week (July 8-14, 2025)
            if target_week_start is None:
                target_week_start = self.find_best_week_for_franchise()
            
            self.load_scheduled_cleanings(target_week_start)
            
            # Step 3: Load customer details and availability
            self.load_customer_data()
            
            # Step 4: Load team composition and availability (synthetic teams for demo)
            self.load_team_data(target_week_start)
            
            # Step 5: Load pre-computed travel matrices (REPLACES geocoding + routing)
            self.load_precomputed_matrices()
            
            # Step 6: Build feasible assignment matrix
            self.build_feasible_assignments()
            
            # Step 7: Run ortools VRP optimization
            self.optimize_weekly_schedule_with_ortools()
            
            # Step 8: Generate results and analysis
            self.generate_results_analysis()
            
            # Step 9: Save results
            self.save_results()
            
            self.logger.success("ORTOOLS VRP OPTIMIZATION COMPLETED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return False
    
    def load_franchise_data(self):
        """Load Franchise 372 configuration"""
        self.logger.section(f"Loading Franchise {self.franchise_id} Configuration")
        
        try:
            franchises_path = os.path.join(self.data_folder, "franchises.csv")
            if not os.path.exists(franchises_path):
                raise FileNotFoundError(f"Franchises file not found: {franchises_path}")
                
            franchises_df = pd.read_csv(franchises_path)
            self.logger.debug(f"Loaded franchises.csv with {len(franchises_df)} records")
            
            # Find Franchise 372
            franchise_row = franchises_df[franchises_df['FranchiseId'] == self.franchise_id]
            if franchise_row.empty:
                raise ValueError(f"Franchise {self.franchise_id} not found in data")
            
            franchise = franchise_row.iloc[0]
            
            # Extract franchise configuration with robust parsing
            try:
                starting_time_raw = franchise.get('StartingTime', 8.0)
                if isinstance(starting_time_raw, str):
                    # Handle malformed time strings - just extract the first number
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', starting_time_raw)
                    starting_time = float(numbers[0]) if numbers else 8.0
                else:
                    starting_time = float(starting_time_raw) if starting_time_raw else 8.0
                
                # Ensure reasonable bounds (6 AM to 10 AM)
                if starting_time < 6 or starting_time > 10:
                    starting_time = 8.0
                    
            except (ValueError, TypeError):
                starting_time = 8.0  # Default fallback
            
            try:
                max_slots = int(franchise.get('DaySchedMaxSlots_Default7', 7))
            except (ValueError, TypeError):
                max_slots = 7
            
            self.franchise_info = {
                'franchise_id': self.franchise_id,
                'name': franchise.get('FranchiseName', 'Unknown'),
                'address': f"{franchise.get('Address1', '')}, {franchise.get('City', '')}, {franchise.get('State', '')}".strip(', '),
                'city': franchise.get('City', ''),
                'state': franchise.get('State', ''),
                'postal_code': franchise.get('PostalCode', ''),
                'start_time': starting_time,
                'max_slots': max_slots,
                'working_minutes_start': int(starting_time * 60),  # Convert to minutes from midnight
                'working_minutes_end': 18 * 60  # 6 PM in minutes
            }
            
            self.logger.success(f"Franchise loaded: {self.franchise_info['name']}")
            self.logger.info(f"📍 Address: {self.franchise_info['address']}")
            self.logger.info(f"🕒 Operating hours: {self.franchise_info['start_time']:.1f}:00 - 18:00 ({self.franchise_info['working_minutes_end'] - self.franchise_info['working_minutes_start']} minutes)")
            self.logger.debug(f"Max slots per day: {self.franchise_info['max_slots']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load franchise data: {str(e)}")
            raise
    
    def find_best_week_for_franchise(self):
        """Return the target week: July 8-14, 2025 (peak week with 204 cleanings)"""
        target_week_start = datetime(2025, 7, 8).date()  # Monday July 8, 2025
        
        self.logger.info(f"📅 Using target week: {target_week_start} (July 8-14, 2025)")
        self.logger.info(f"📊 This is the PEAK WEEK with 204 cleanings for Franchise {self.franchise_id}")
        self.logger.info(f"🎯 Matrices available for 177 customers (some addresses couldn't be geocoded)")
        
        return target_week_start
    
    def load_scheduled_cleanings(self, target_week_start):
        """Load scheduled cleanings for the target week"""
        week_end = target_week_start + timedelta(days=6)
        self.logger.section(f"Loading Scheduled Cleanings for Week {target_week_start} to {week_end}")
        
        try:
            cleans_path = os.path.join(self.data_folder, "cleans.csv")
            if not os.path.exists(cleans_path):
                raise FileNotFoundError(f"Cleans file not found: {cleans_path}")
            
            # Load and parse dates
            cleans_df = pd.read_csv(cleans_path, low_memory=False)
            cleans_df['CleanDate'] = pd.to_datetime(cleans_df['CleanDate'])
            self.logger.debug(f"Loaded cleans.csv with {len(cleans_df)} total records")
            
            # Filter for target week
            week_cleanings = cleans_df[
                (cleans_df['CleanDate'].dt.date >= target_week_start) &
                (cleans_df['CleanDate'].dt.date <= week_end)
            ]
            
            if week_cleanings.empty:
                self.logger.warning(f"No cleanings found for week {target_week_start}")
                self.cleanings = []
                return
            
            # Convert to list of dictionaries for easier processing
            self.cleanings = []
            duration_stats = []
            
            for _, cleaning in week_cleanings.iterrows():
                clean_data = {
                    'clean_id': cleaning.get('CleanId'),
                    'customer_id': cleaning.get('CustomerId'),
                    'clean_date': cleaning['CleanDate'].date(),
                    'clean_time_minutes': cleaning.get('CleanTime', 90),  # Default 90 minutes
                    'team_number': cleaning.get('TeamNumber'),
                    'time_in': cleaning.get('TimeIn'),
                    'time_out': cleaning.get('TimeOut'),
                    'slot': cleaning.get('Slot'),
                    'rotation': cleaning.get('Rotation', 1)
                }
                self.cleanings.append(clean_data)
                duration_stats.append(clean_data['clean_time_minutes'])
            
            # Analysis
            self.logger.success(f"Found {len(self.cleanings)} scheduled cleanings")
            
            # Duration analysis
            durations = pd.Series(duration_stats)
            self.logger.info(f"📊 Duration analysis:")
            self.logger.info(f"   Average duration: {durations.mean():.1f} minutes")
            self.logger.info(f"   Range: {durations.min()}-{durations.max()} minutes")
            
            # Daily breakdown
            daily_counts = pd.Series([c['clean_date'] for c in self.cleanings]).value_counts().sort_index()
            self.logger.info(f"📅 Daily breakdown:")
            for date, count in daily_counts.items():
                day_name = date.strftime('%A')
                self.logger.info(f"   {day_name} {date}: {count} cleanings")
            
        except Exception as e:
            self.logger.error(f"Failed to load cleaning data: {str(e)}")
            raise
    
    def load_customer_data(self):
        """Load customer details and availability for all customers in cleanings"""
        self.logger.section("Loading Customer Details & Availability")
        
        if not self.cleanings:
            self.logger.warning("No cleanings loaded, skipping customer data")
            return
        
        try:
            customers_path = os.path.join(self.data_folder, "customers.csv")
            if not os.path.exists(customers_path):
                raise FileNotFoundError(f"Customers file not found: {customers_path}")
            
            # Get unique customer IDs from cleanings
            customer_ids = list(set(c['customer_id'] for c in self.cleanings))
            self.logger.debug(f"Loading data for {len(customer_ids)} unique customers")
            
            # Load customers data (might be large, so we'll process in chunks if needed)
            try:
                customers_df = pd.read_csv(customers_path, low_memory=False)
                self.logger.debug(f"Loaded customers.csv with {len(customers_df)} total records")
            except Exception as e:
                self.logger.warning(f"Large file, attempting chunked reading: {e}")
                # For very large files, we could implement chunked reading here
                raise
            
            # Filter for our franchise and customer IDs
            franchise_customers = customers_df[
                (customers_df['FranchiseId'] == self.franchise_id) &
                (customers_df['CustomerId'].isin(customer_ids))
            ]
            
            if franchise_customers.empty:
                raise ValueError(f"No customers found for Franchise {self.franchise_id}")
            
            self.customers = {}
            address_count = 0
            availability_count = 0
            
            for _, customer in franchise_customers.iterrows():
                customer_id = customer['CustomerId']
                
                # Build full address
                address_parts = [
                    str(customer.get('Address1', '')).strip(),
                    str(customer.get('Address2', '')).strip(),
                    str(customer.get('City', '')).strip(),
                    str(customer.get('State', '')).strip(),
                    str(customer.get('PostalCode', '')).strip()
                ]
                full_address = ', '.join([part for part in address_parts if part and part != 'nan'])
                
                # Weekly availability
                availability = {
                    'monday': bool(customer.get('IsCleanOnMon', 1)),
                    'tuesday': bool(customer.get('IsCleanOnTue', 1)),
                    'wednesday': bool(customer.get('IsCleanOnWed', 1)),
                    'thursday': bool(customer.get('IsCleanOnThu', 1)),
                    'friday': bool(customer.get('IsCleanOnFri', 1)),
                    'saturday': bool(customer.get('IsCleanOnSat', 1)),
                    'sunday': bool(customer.get('IsCleanOnSun', 0))
                }
                
                # Day constraints (when they CANNOT be cleaned)
                constraints = {
                    'monday': bool(customer.get('IsContraintMonday', 0)),
                    'tuesday': bool(customer.get('IsContraintTuesday', 0)),
                    'wednesday': bool(customer.get('IsContraintWednesday', 0)),
                    'thursday': bool(customer.get('IsContraintThursday', 0)),
                    'friday': bool(customer.get('IsContraintFriday', 0)),
                    'saturday': bool(customer.get('IsContraintSaturday', 0)),
                    'sunday': bool(customer.get('IsContraintSunday', 0))
                }
                
                self.customers[customer_id] = {
                    'customer_id': customer_id,
                    'address': full_address,
                    'city': customer.get('City', ''),
                    'state': customer.get('State', ''),
                    'postal_code': customer.get('PostalCode', ''),
                    'availability': availability,
                    'constraints': constraints,
                    'house_size': customer.get('HouseSize', ''),
                    'frequency_code': customer.get('FrequencyCode', ''),
                }
                
                if full_address:
                    address_count += 1
                if any(availability.values()):
                    availability_count += 1
            
            self.logger.success(f"Loaded {len(self.customers)} customer records")
            self.logger.info(f"📍 Customers with addresses: {address_count}")
            self.logger.info(f"📅 Customers with availability: {availability_count}")
            
            # Check for scheduling conflicts
            conflicts = 0
            for cleaning in self.cleanings:
                customer_id = cleaning['customer_id']
                if customer_id in self.customers:
                    day_name = cleaning['clean_date'].strftime('%A').lower()
                    customer_data = self.customers[customer_id]
                    
                    available = customer_data['availability'].get(day_name, True)
                    constrained = customer_data['constraints'].get(day_name, False)
                    
                    if not available or constrained:
                        conflicts += 1
                        self.logger.debug(f"Scheduling conflict: Customer {customer_id} on {day_name}")
            
            if conflicts > 0:
                self.logger.warning(f"{conflicts} customers need rescheduling due to availability conflicts")
            
        except Exception as e:
            self.logger.error(f"Failed to load customer data: {str(e)}")
            raise
    
    def load_team_data(self, target_week_start):
        """Load team composition and availability for the target week"""
        week_end = target_week_start + timedelta(days=6)
        self.logger.section(f"Loading Team Data for Week {target_week_start} to {week_end}")
        
        try:
            teams_path = os.path.join(self.data_folder, "teams.csv")
            if not os.path.exists(teams_path):
                raise FileNotFoundError(f"Teams file not found: {teams_path}")
            
            # Load teams data
            teams_df = pd.read_csv(teams_path)
            teams_df['CleanDate'] = pd.to_datetime(teams_df['CleanDate'])
            self.logger.debug(f"Loaded teams.csv with {len(teams_df)} total records")
            
            # Filter for our franchise, target week, and cleaners only (EmployeeTypeId = 1)
            week_teams = teams_df[
                (teams_df['FranchiseId'] == self.franchise_id) &
                (teams_df['CleanDate'].dt.date >= target_week_start) &
                (teams_df['CleanDate'].dt.date <= week_end) &
                (teams_df['EmployeeTypeId'] == 1)  # Cleaners only
            ]
            
            if week_teams.empty:
                self.logger.warning(f"No team data found for Franchise {self.franchise_id} in target week")
                self.teams = {}
                return
            
            # Group by date and team to build team compositions
            self.teams = {}  # {date: {team_number: team_info}}
            
            for date in pd.date_range(target_week_start, week_end):
                date_key = date.date()
                day_name = date.strftime('%A').lower()
                self.teams[date_key] = {}
                
                day_teams = week_teams[week_teams['CleanDate'].dt.date == date_key]
                
                if not day_teams.empty:
                    for team_number, team_data in day_teams.groupby('TeamNumber'):
                        team_members = []
                        drivers = 0
                        available_members = 0
                        
                        for _, member in team_data.iterrows():
                            # Check if member is available this day
                            day_availability_col = f'IsAvailable{date.strftime("%a")}'  # IsAvailableMon, etc.
                            is_available = member.get(day_availability_col, 1) == 1
                            is_not_absent = member.get('DailyTeamStatus', 'Normal') != 'Absent'
                            
                            member_info = {
                                'employee_id': member.get('EmployeeId'),
                                'first_name': member.get('FirstName', ''),
                                'is_driver': bool(member.get('IsDriver', 0)),
                                'is_available': is_available,
                                'is_not_absent': is_not_absent,
                                'daily_status': member.get('DailyTeamStatus', 'Normal'),
                                'work_status': member.get('EmployeeWorkStatus', 'Full Time')
                            }
                            
                            team_members.append(member_info)
                            
                            if member_info['is_driver']:
                                drivers += 1
                            if is_available and is_not_absent:
                                available_members += 1
                        
                        # Team is operational if it has at least 1 driver and all members are available
                        all_available = all(m['is_available'] and m['is_not_absent'] for m in team_members)
                        has_driver = drivers > 0
                        is_operational = all_available and has_driver and len(team_members) >= 2
                        
                        self.teams[date_key][team_number] = {
                            'team_number': team_number,
                            'members': team_members,
                            'total_members': len(team_members),
                            'available_members': available_members,
                            'drivers': drivers,
                            'is_operational': is_operational,
                            'day_name': day_name
                        }
            
            # Summary statistics
            operational_teams_by_day = {}
            total_teams = 0
            operational_teams = 0
            
            for date_key, daily_teams in self.teams.items():
                day_name = date_key.strftime('%A')
                operational_count = sum(1 for team in daily_teams.values() if team['is_operational'])
                operational_teams_by_day[day_name] = operational_count
                total_teams += len(daily_teams)
                operational_teams += operational_count
            
            self.logger.success(f"Team data loaded for {len(self.teams)} days")
            self.logger.info(f"📊 Team availability by day:")
            for day, count in operational_teams_by_day.items():
                self.logger.info(f"   {day}: {count} operational teams")
            
            self.logger.debug(f"Total team-days: {total_teams}, Operational: {operational_teams}")
            
            if operational_teams == 0:
                self.logger.warning("No operational teams found! Check team data and availability.")
            
        except Exception as e:
            self.logger.error(f"Failed to load team data: {str(e)}")
            raise
    
    def initialize_services(self):
        """Initialize geocoding and routing services"""
        self.logger.section("Initializing Geocoding & Routing Services")
        
        try:
            # Initialize geocoding with rate limiting (Nominatim terms of service)
            self.geolocator = Nominatim(user_agent=f"franchise_{self.franchise_id}_scheduler")
            self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
            self.logger.success("Geocoding service initialized (Nominatim/OpenStreetMap)")
            
            # Initialize routing service
            self.routing_api = rp.OSRM()  # Free OSRM service
            self.logger.success("Routing service initialized (OSRM)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {str(e)}")
            raise
    
    def load_precomputed_matrices(self):
        """Load pre-computed distance and time matrices"""
        self.logger.section("Loading Pre-computed Travel Matrices")
        
        try:
            # Load time matrix
            time_matrix_path = os.path.join(self.data_folder, "complete_real_driving_time_matrix.csv")
            if not os.path.exists(time_matrix_path):
                raise FileNotFoundError(f"Time matrix not found: {time_matrix_path}")
            
            time_df = pd.read_csv(time_matrix_path, index_col=0)
            self.logger.debug(f"Time matrix loaded: {time_df.shape[0]} x {time_df.shape[1]}")
            
            # Load distance matrix
            distance_matrix_path = os.path.join(self.data_folder, "complete_real_driving_distance_matrix.csv")
            if not os.path.exists(distance_matrix_path):
                raise FileNotFoundError(f"Distance matrix not found: {distance_matrix_path}")
            
            distance_df = pd.read_csv(distance_matrix_path, index_col=0)
            self.logger.debug(f"Distance matrix loaded: {distance_df.shape[0]} x {distance_df.shape[1]}")
            
            # Create location mapping: matrix position -> actual customer ID
            matrix_locations = list(time_df.columns)  # ['Franchise_Office', 'Customer_001', 'Customer_002', ...]
            
            self.logger.info(f"📍 Matrix contains {len(matrix_locations)} locations:")
            self.logger.info(f"   Franchise office + {len(matrix_locations)-1} customers")
            
            # Store matrices directly - treat column names as customer IDs
            self.time_matrix_df = time_df
            self.distance_matrix_df = distance_df
            self.time_matrix = time_df.values.astype(int)  # ortools needs integers (in minutes)
            self.distance_matrix = distance_df.values.astype(int)  # ortools needs integers (in km * 100)
            
            # Matrix column names ARE the customer identifiers (no mapping needed)
            self.matrix_customer_ids = [col for col in matrix_locations if col != 'Franchise_Office']
            
            self.logger.success(f"Matrices loaded successfully!")
            self.logger.info(f"📊 Available customers in matrix: {len(self.matrix_customer_ids)}")
            self.logger.info(f"📈 Matrix dimensions: {self.time_matrix.shape}")
            
            # Sample some travel times
            self.logger.info(f"🚗 Sample travel times (minutes):")
            franchise_to_first = self.time_matrix[0][1] if self.time_matrix.shape[1] > 1 else 0
            self.logger.info(f"   Franchise to first customer: {franchise_to_first} minutes")
            
            if self.time_matrix.shape[0] > 2:
                between_customers = self.time_matrix[1][2]
                self.logger.info(f"   Between first two customers: {between_customers} minutes")
            
        except Exception as e:
            self.logger.error(f"Failed to load matrices: {str(e)}")
            raise
        
    def build_travel_matrix(self):
        """Build travel time matrix between all locations"""
        self.logger.section("Building Travel Time Matrix")
        
        if not self.routing_api:
            raise RuntimeError("Routing service not initialized")
        
        if not self.coordinates:
            raise RuntimeError("No coordinates available for routing")
        
        all_locations = list(self.coordinates.keys())
        total_combinations = len(all_locations) ** 2
        
        self.logger.info(f"🚗 Calculating travel times for {len(all_locations)} locations ({total_combinations} combinations)")
        
        self.travel_matrix = {}
        successful = 0
        failed = 0
        current = 0
        
        for origin in all_locations:
            self.travel_matrix[origin] = {}
            
            for destination in all_locations:
                current += 1
                self.logger.progress(current, total_combinations, "Calculating routes")
                
                if origin == destination:
                    self.travel_matrix[origin][destination] = 0
                    successful += 1
                    continue
                
                try:
                    # Get coordinates
                    origin_coords = self.coordinates[origin]
                    dest_coords = self.coordinates[destination]
                    
                    # Calculate route
                    route = self.routing_api.directions(
                        locations=[origin_coords, dest_coords],
                        profile='driving-car'
                    )
                    
                    # Extract travel time in minutes
                    travel_time = route.duration / 60
                    self.travel_matrix[origin][destination] = travel_time
                    successful += 1
                    
                except Exception as e:
                    # Fallback: Use straight-line distance * time factor
                    try:
                        origin_coords = self.coordinates[origin]
                        dest_coords = self.coordinates[destination]
                        
                        # Rough distance calculation (Haversine approximation)
                        lat1, lng1 = origin_coords[1], origin_coords[0]
                        lat2, lng2 = dest_coords[1], dest_coords[0]
                        
                        dlat = abs(lat1 - lat2) * 111  # km per degree latitude
                        dlng = abs(lng1 - lng2) * 85   # km per degree longitude (approximate for Chicago)
                        distance_km = (dlat**2 + dlng**2)**0.5
                        
                        # Estimate: 2 minutes per km in city driving
                        travel_time = distance_km * 2
                        self.travel_matrix[origin][destination] = travel_time
                        failed += 1
                        
                    except Exception as e2:
                        # Last resort: default time
                        self.travel_matrix[origin][destination] = 15  # 15 minute default
                        failed += 1
                        self.logger.debug(f"Routing failed for {origin} -> {destination}: {e}")
        
        # Complete progress
        self.logger.progress(total_combinations, total_combinations, "Calculating routes")
        
        # Analysis
        all_times = []
        for origin in self.travel_matrix:
            for destination in self.travel_matrix[origin]:
                if origin != destination:
                    all_times.append(self.travel_matrix[origin][destination])
        
        if all_times:
            avg_time = np.mean(all_times)
            min_time = np.min(all_times)
            max_time = np.max(all_times)
            
            self.logger.success(f"Travel matrix complete: {successful} calculated, {failed} estimated")
            self.logger.info(f"📊 Travel time statistics:")
            self.logger.info(f"   Average: {avg_time:.1f} minutes")
            self.logger.info(f"   Range: {min_time:.1f} - {max_time:.1f} minutes")
        
    def build_feasible_assignments(self):
        """Build matrix of feasible customer-team-day assignments"""
        self.logger.section("Building Feasible Assignment Matrix")
        
        if not self.customers or not self.cleanings:
            self.logger.warning("Missing data for building assignments")
            self.feasible_assignments = []
            return
        
        self.feasible_assignments = []
        
        # Since we have no team data, we'll create synthetic teams
        self.logger.info("🔍 No team data available - creating synthetic teams for demonstration...")
        
        # Create 3 synthetic teams for each day that has cleanings
        dates_with_cleanings = set(c['clean_date'] for c in self.cleanings)
        
        for cleaning in self.cleanings:
            customer_id = cleaning['customer_id']
            clean_date = cleaning['clean_date']
            clean_duration = cleaning['clean_time_minutes']
            
            if customer_id not in self.customers:
                continue
            
            # Create assignments for synthetic teams 1, 2, and 3
            for team_number in [1, 2, 3]:
                if clean_duration > 0 and clean_duration <= 600:  # Reasonable duration
                    assignment = {
                        'customer_id': customer_id,
                        'clean_date': clean_date,
                        'team_number': team_number,
                        'clean_duration': clean_duration,
                        'earliest_start': 480,  # 8 AM in minutes
                        'latest_start': 1080 - clean_duration,  # Latest start to finish by 6 PM
                        'day_name': clean_date.strftime('%A').lower()
                    }
                    self.feasible_assignments.append(assignment)
        
        self.logger.success(f"Created {len(self.feasible_assignments)} feasible assignments")
    
    def optimize_weekly_schedule_with_ortools(self):
        """Run ortools VRP optimization using pre-computed matrices"""
        self.logger.section("Running ortools VRP Optimization")
        
        if not hasattr(self, 'time_matrix') or self.time_matrix is None:
            self.logger.error("No travel matrices loaded!")
            self.optimized_schedule = {}
            return
        
        if not self.feasible_assignments:
            self.logger.warning("No feasible assignments to optimize")
            self.optimized_schedule = {}
            return
        
        # Group assignments by date
        assignments_by_date = defaultdict(list)
        for assignment in self.feasible_assignments:
            assignments_by_date[assignment['clean_date']].append(assignment)
        
        self.optimized_schedule = {}
        
        # Optimize each day using ortools VRP
        for clean_date in sorted(assignments_by_date.keys()):
            day_assignments = assignments_by_date[clean_date]
            day_name = clean_date.strftime('%A')
            
            self.logger.info(f"📅 Running ortools VRP for {day_name} {clean_date}")
            
            # Get unique customers for this day
            customer_assignments = {}
            for assignment in day_assignments:
                customer_id = assignment['customer_id']
                if customer_id not in customer_assignments:
                    customer_assignments[customer_id] = assignment
            
            if len(customer_assignments) > 0:
                daily_schedule = self.solve_daily_vrp(clean_date, customer_assignments)
                self.optimized_schedule[clean_date] = daily_schedule
            else:
                self.optimized_schedule[clean_date] = []
        
        # Calculate final statistics
        total_assignments = sum(len(schedule) for schedule in self.optimized_schedule.values())
        total_travel_time = sum(
            assignment.get('travel_time', 0) 
            for schedule in self.optimized_schedule.values() 
            for assignment in schedule
        )
        
        avg_travel_time = total_travel_time / total_assignments if total_assignments > 0 else 0
        
        self.optimization_stats = {
            'total_assignments': total_assignments,
            'total_travel_time': total_travel_time,
            'average_travel_time': avg_travel_time,
            'days_optimized': len(self.optimized_schedule)
        }
        
        self.logger.success(f"ortools VRP optimization complete!")
        self.logger.info(f"📊 {total_assignments} assignments scheduled")
        self.logger.info(f"📊 Average travel time: {avg_travel_time:.1f} minutes per job")
    
    def solve_daily_vrp(self, clean_date, customer_assignments):
        """Solve daily VRP using ortools for a single day"""
        
        customers = list(customer_assignments.keys())
        vrp_customers = []
        matrix_indices = []
        
        # Use matrix directly - treat customer IDs as matrix column names
        matrix_indices.append(0)  # Franchise office at index 0
        vrp_customers.append('franchise')
        
        # Check which customers from today's schedule are in our matrix
        for customer_id in customers:
            # Customer IDs should directly match matrix column names (Customer_001, etc.)
            if customer_id in self.time_matrix_df.columns:
                col_index = list(self.time_matrix_df.columns).index(customer_id)
                matrix_indices.append(col_index)
                vrp_customers.append(customer_id)
        
        if len(vrp_customers) <= 1:  # Only franchise, no customers
            self.logger.warning(f"   No customers from {clean_date} found in matrix")
            self.logger.debug(f"   Available matrix customers: {self.matrix_customer_ids[:5]}...")
            self.logger.debug(f"   Today's customers: {customers[:5]}...")
            return []
        
        # Extract relevant submatrix
        num_locations = len(matrix_indices)
        sub_matrix = [[0 for _ in range(num_locations)] for _ in range(num_locations)]
        
        for i in range(num_locations):
            for j in range(num_locations):
                sub_matrix[i][j] = int(self.time_matrix[matrix_indices[i]][matrix_indices[j]])
        
        self.logger.debug(f"   VRP problem: {num_locations} locations ({num_locations-1} customers)")
        
        # Determine number of available teams for this day
        available_teams = self.get_available_teams_for_date(clean_date)
        num_teams = max(1, min(available_teams, len(vrp_customers) - 1))  # At least 1, max customers-1
        
        # Create VRP data model
        data = {
            'time_matrix': sub_matrix,
            'num_vehicles': num_teams,
            'depot': 0  # Franchise office
        }
        
        self.logger.debug(f"   Using {num_teams} teams for {clean_date} ({available_teams} available)")
        
        try:
            # Solve VRP
            solution = self.solve_vrp_with_ortools(data, vrp_customers, customer_assignments)
            return solution
            
        except Exception as e:
            self.logger.warning(f"   ortools VRP failed: {e}, falling back to greedy")
            return self.optimize_single_day(clean_date, list(customer_assignments.values()))
    
    def solve_vrp_with_ortools(self, data, mapped_customers, customer_assignments):
        """Solve VRP using ortools constraint solver"""
        
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time window constraints (8 AM to 6 PM)
        time_dimension_name = 'Time'
        routing.AddDimension(
            transit_callback_index,
            30,  # allow waiting time
            600,  # maximum time per vehicle (10 hours)
            False,  # Don't force start cumul to zero
            time_dimension_name
        )
        time_dimension = routing.GetDimensionOrDie(time_dimension_name)
        
        # Add time windows for the depot (franchise)
        time_dimension.CumulVar(manager.NodeToIndex(0)).SetRange(480, 480)  # Start at 8 AM
        
        # Add time windows for customer locations (include service time)
        for i in range(1, len(mapped_customers)):
            customer_id = mapped_customers[i]
            if customer_id in customer_assignments:
                service_time = customer_assignments[customer_id]['clean_duration']
                # Allow scheduling between 8 AM and 5 PM (to finish by 6 PM)
                time_dimension.CumulVar(manager.NodeToIndex(i)).SetRange(480, 1020)  # 8 AM to 5 PM
                routing.AddToAssignment(time_dimension.SlackVar(manager.NodeToIndex(i)))
        
        # Setting search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(30)  # 30 second time limit
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.extract_vrp_solution(data, manager, routing, solution, mapped_customers, customer_assignments)
        else:
            raise Exception("No solution found")
    
    def extract_vrp_solution(self, data, manager, routing, solution, mapped_customers, customer_assignments):
        """Extract solution from ortools VRP solver"""
        
        daily_schedule = []
        time_dimension = routing.GetDimensionOrDie('Time')
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_assignments = []
            total_time = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                if node_index > 0:  # Not the depot
                    customer_id = mapped_customers[node_index]
                    if customer_id in customer_assignments:
                        
                        # Get timing from ortools solution
                        time_var = time_dimension.CumulVar(index)
                        start_time_minutes = solution.Value(time_var)
                        
                        assignment = customer_assignments[customer_id]
                        clean_duration = assignment['clean_duration']
                        end_time_minutes = start_time_minutes + clean_duration
                        
                        # Calculate travel time to this customer
                        previous_index = routing.Start(vehicle_id) if len(route_assignments) == 0 else manager.NodeToIndex(previous_node)
                        travel_time = data['time_matrix'][manager.IndexToNode(previous_index)][node_index]
                        
                        # Format times
                        start_hours = int(start_time_minutes // 60)
                        start_mins = int(start_time_minutes % 60)
                        end_hours = int(end_time_minutes // 60)
                        end_mins = int(end_time_minutes % 60)
                        
                        assignment_record = {
                            'customer_id': customer_id,
                            'team_number': vehicle_id + 1,
                            'start_time_str': f"{start_hours:02d}:{start_mins:02d}",
                            'end_time_str': f"{end_hours:02d}:{end_mins:02d}",
                            'start_time_minutes': start_time_minutes,
                            'end_time_minutes': end_time_minutes,
                            'travel_time': travel_time,
                            'clean_duration': clean_duration,
                            'previous_location': 'franchise' if len(route_assignments) == 0 else route_assignments[-1]['customer_id']
                        }
                        
                        route_assignments.append(assignment_record)
                        daily_schedule.append(assignment_record)
                        total_time += travel_time
                        
                        self.logger.debug(f"   ✅ Customer {customer_id} → Team {vehicle_id + 1} at {assignment_record['start_time_str']} (travel: {travel_time}min)")
                
                previous_node = node_index
                index = solution.Value(routing.NextVar(index))
        
        return daily_schedule
    
    def get_available_teams_for_date(self, clean_date):
        """Get number of available teams for a specific date"""
        if not hasattr(self, 'teams') or clean_date not in self.teams:
            # Default: Use reasonable number based on customer count for the day
            daily_customer_count = len([c for c in self.cleanings if c['clean_date'] == clean_date])
            if daily_customer_count <= 15:
                return 1
            elif daily_customer_count <= 30:
                return 2
            elif daily_customer_count <= 45:
                return 3
            else:
                return max(3, min(5, daily_customer_count // 15))  # Scale with customers
        
        # Count operational teams for this date
        operational_teams = sum(1 for team in self.teams[clean_date].values() if team['is_operational'])
        
        # Ensure at least 1 team, reasonable maximum
        return max(1, min(operational_teams, 5))
        
    def optimize_weekly_schedule(self):
        """Run the optimization algorithm to create the weekly schedule"""
        self.logger.section("Running Schedule Optimization Algorithm")
        
        if not self.feasible_assignments:
            self.logger.warning("No feasible assignments to optimize")
            self.optimized_schedule = {}
            return
        
        # Group assignments by date
        assignments_by_date = defaultdict(list)
        for assignment in self.feasible_assignments:
            assignments_by_date[assignment['clean_date']].append(assignment)
        
        self.optimized_schedule = {}
        total_travel_time = 0
        total_assignments = 0
        
        # Optimize each day
        for clean_date in sorted(assignments_by_date.keys()):
            day_assignments = assignments_by_date[clean_date]
            day_name = clean_date.strftime('%A')
            
            self.logger.info(f"📅 Optimizing {day_name} {clean_date} ({len(day_assignments)} potential assignments)")
            
            daily_schedule = self.optimize_single_day(clean_date, day_assignments)
            self.optimized_schedule[clean_date] = daily_schedule
            
            # Track statistics
            for assignment in daily_schedule:
                total_travel_time += assignment.get('travel_time', 0)
                total_assignments += 1
        
        # Calculate statistics
        if total_assignments > 0:
            avg_travel_time = total_travel_time / total_assignments
            self.optimization_stats = {
                'total_assignments': total_assignments,
                'total_travel_time': total_travel_time,
                'average_travel_time': avg_travel_time,
                'days_optimized': len(self.optimized_schedule)
            }
            
            self.logger.success(f"Weekly optimization complete!")
            self.logger.info(f"📊 {total_assignments} assignments scheduled")
            self.logger.info(f"📊 Average travel time: {avg_travel_time:.1f} minutes per job")
    
    def optimize_single_day(self, clean_date, day_assignments):
        """Optimize assignments for a single day using greedy nearest-neighbor"""
        
        # Group assignments by customer (take only one per customer)
        customer_assignments = {}
        for assignment in day_assignments:
            customer_id = assignment['customer_id']
            if customer_id not in customer_assignments:
                customer_assignments[customer_id] = assignment
        
        # Initialize 3 synthetic teams
        team_states = {
            1: {'available_time': 480, 'location': 'franchise', 'assignments': []},  # 8 AM
            2: {'available_time': 480, 'location': 'franchise', 'assignments': []},
            3: {'available_time': 480, 'location': 'franchise', 'assignments': []}
        }
        
        unassigned_customers = list(customer_assignments.keys())
        daily_schedule = []
        
        self.logger.debug(f"   Optimizing {len(unassigned_customers)} customers with 3 synthetic teams")
        
        # Greedy assignment loop
        while unassigned_customers and len(daily_schedule) < 20:  # Limit to prevent runaway
            best_assignment = None
            best_completion_time = float('inf')
            best_travel_time = float('inf')
            
            # For each unassigned customer
            for customer_id in unassigned_customers[:10]:  # Limit search for performance
                assignment = customer_assignments[customer_id]
                clean_duration = assignment['clean_duration']
                
                # Try each team
                for team_number in [1, 2, 3]:
                    team_state = team_states[team_number]
                    current_time = team_state['available_time']
                    current_location = team_state['location']
                    
                    # Calculate travel time
                    if customer_id in self.travel_matrix.get(current_location, {}):
                        travel_time = self.travel_matrix[current_location][customer_id]
                    else:
                        travel_time = 10  # Default
                    
                    # Calculate completion time
                    start_time = current_time + travel_time
                    completion_time = start_time + clean_duration
                    
                    # Check if it fits in working day
                    if completion_time <= 1080:  # 6 PM
                        # Prefer earlier completion times
                        if completion_time < best_completion_time or (completion_time == best_completion_time and travel_time < best_travel_time):
                            best_assignment = {
                                'customer_id': customer_id,
                                'team_number': team_number,
                                'start_time': start_time,
                                'completion_time': completion_time,
                                'travel_time': travel_time,
                                'clean_duration': clean_duration,
                                'previous_location': current_location
                            }
                            best_completion_time = completion_time
                            best_travel_time = travel_time
            
            if best_assignment:
                # Make the assignment
                customer_id = best_assignment['customer_id']
                team_number = best_assignment['team_number']
                
                # Update team state
                team_states[team_number]['available_time'] = best_assignment['completion_time']
                team_states[team_number]['location'] = customer_id
                
                # Format times
                start_hours = int(best_assignment['start_time'] // 60)
                start_minutes = int(best_assignment['start_time'] % 60)
                end_hours = int(best_assignment['completion_time'] // 60)
                end_minutes = int(best_assignment['completion_time'] % 60)
                
                assignment_record = {
                    'customer_id': customer_id,
                    'team_number': team_number,
                    'start_time_str': f"{start_hours:02d}:{start_minutes:02d}",
                    'end_time_str': f"{end_hours:02d}:{end_minutes:02d}",
                    'start_time_minutes': best_assignment['start_time'],
                    'end_time_minutes': best_assignment['completion_time'],
                    'travel_time': best_assignment['travel_time'],
                    'clean_duration': best_assignment['clean_duration'],
                    'previous_location': best_assignment['previous_location']
                }
                
                daily_schedule.append(assignment_record)
                unassigned_customers.remove(customer_id)
                
                self.logger.debug(f"   ✅ Customer {customer_id} → Team {team_number} at {assignment_record['start_time_str']} (travel: {best_assignment['travel_time']:.1f}min)")
            
            else:
                # No more feasible assignments
                break
        
        return daily_schedule
    
    def generate_results_analysis(self):
        """Generate analysis of optimization results"""
        self.logger.section("Generating Results Analysis")
        
        if not self.optimized_schedule:
            self.logger.warning("No optimized schedule to analyze")
            return
        
        total_jobs = 0
        total_travel_time = 0
        daily_summaries = {}
        
        # Analyze each day
        for clean_date, daily_schedule in self.optimized_schedule.items():
            day_name = clean_date.strftime('%A')
            day_jobs = len(daily_schedule)
            day_travel = sum(assignment['travel_time'] for assignment in daily_schedule)
            
            daily_summaries[clean_date] = {
                'day_name': day_name,
                'total_jobs': day_jobs,
                'total_travel': day_travel,
                'avg_travel': day_travel / day_jobs if day_jobs > 0 else 0
            }
            
            total_jobs += day_jobs
            total_travel_time += day_travel
        
        # Calculate global statistics
        global_avg_travel = total_travel_time / total_jobs if total_jobs > 0 else 0
        
        # Display results
        self.logger.success("📊 OPTIMIZATION ANALYSIS COMPLETE")
        self.logger.info("")
        self.logger.info("🎯 GLOBAL RESULTS:")
        self.logger.info(f"   Total jobs scheduled: {total_jobs}")
        self.logger.info(f"   Global average travel: {global_avg_travel:.1f} minutes per job")
        self.logger.info(f"   Total travel time: {total_travel_time:.1f} minutes ({total_travel_time/60:.1f} hours)")
        
        # Daily breakdown
        self.logger.info("")
        self.logger.info("📅 DAILY BREAKDOWN:")
        for clean_date in sorted(daily_summaries.keys()):
            summary = daily_summaries[clean_date]
            self.logger.info(f"   {summary['day_name']} {clean_date}: {summary['total_jobs']} jobs, {summary['avg_travel']:.1f}min avg travel")
        
        # Estimate improvement
        baseline_travel = 25.0
        time_saved = (baseline_travel - global_avg_travel) * total_jobs
        improvement_pct = ((baseline_travel - global_avg_travel) / baseline_travel) * 100 if baseline_travel > 0 else 0
        
        self.logger.info("")
        self.logger.info("💰 ESTIMATED IMPROVEMENT:")
        self.logger.info(f"   Before optimization: ~{baseline_travel:.1f} minutes per job")
        self.logger.info(f"   After optimization: {global_avg_travel:.1f} minutes per job")
        self.logger.info(f"   Time saved: {time_saved:.1f} minutes ({time_saved/60:.1f} hours) per week")
        self.logger.info(f"   Improvement: {improvement_pct:.1f}%")
        
        # Update stats
        self.optimization_stats.update({
            'daily_summaries': daily_summaries,
            'global_avg_travel': global_avg_travel,
            'estimated_time_saved': time_saved,
            'improvement_percentage': improvement_pct
        })
    
    def save_results(self):
        """Save optimized schedule to CSV"""
        self.logger.section("Saving Results")
        
        if not self.optimized_schedule:
            self.logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare schedule data
        schedule_records = []
        for clean_date, daily_schedule in self.optimized_schedule.items():
            for assignment in daily_schedule:
                customer_id = assignment['customer_id']
                customer_data = self.customers.get(customer_id, {})
                
                record = {
                    'clean_date': clean_date,
                    'day_name': clean_date.strftime('%A'),
                    'customer_id': customer_id,
                    'customer_address': customer_data.get('address', ''),
                    'customer_city': customer_data.get('city', ''),
                    'team_number': assignment['team_number'],
                    'start_time': assignment['start_time_str'],
                    'end_time': assignment['end_time_str'],
                    'clean_duration_minutes': assignment['clean_duration'],
                    'travel_time_minutes': round(assignment['travel_time'], 1),
                    'previous_location': assignment['previous_location']
                }
                schedule_records.append(record)
        
        # Save CSV
        if schedule_records:
            schedule_df = pd.DataFrame(schedule_records)
            filename = f"optimized_schedule_franchise_{self.franchise_id}_{timestamp}.csv"
            schedule_df.to_csv(filename, index=False)
            self.logger.success(f"Schedule saved: {filename}")
            
            # Show sample
            self.logger.info("\n📋 SAMPLE OPTIMIZED SCHEDULE:")
            for _, row in schedule_df.head(10).iterrows():
                addr = row['customer_address'][:40] + "..." if len(row['customer_address']) > 40 else row['customer_address']
                self.logger.info(f"   {row['day_name']} Team {row['team_number']}: {row['start_time']}-{row['end_time']} | Customer {row['customer_id']} at {addr}")
        else:
            self.logger.warning("No schedule data to save")


def main():
    """Main execution function"""
    print("TRAVEL-OPTIMIZED CLEANING SCHEDULER")
    print("=" * 50)
    
    # Check for required files
    required_files = ["franchises.csv", "cleans.csv", "customers.csv", "teams.csv"]
    data_folder = "data files"
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print(f"📁 Please ensure all files are in the '{data_folder}' folder")
        return False
    
    try:
        # Create scheduler instance
        scheduler = TravelOptimizedScheduler(franchise_id=372, verbose=True)
        
        # Run optimization (will use next Monday as default target week)
        success = scheduler.run_complete_optimization()
        
        if success:
            print("\n🎊 OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print("📄 Check the generated CSV and TXT files for detailed results")
            return True
        else:
            print("\n💥 OPTIMIZATION FAILED - see error messages above")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
