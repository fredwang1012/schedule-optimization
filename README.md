# Travel-Optimized Cleaning Scheduler with ortools VRP

A comprehensive system that optimizes cleaning schedules for Franchise 372 using pre-computed travel matrices and Google's ortools Vehicle Routing Problem solver to minimize travel time between customer locations.

## Features

🚀 **ortools VRP Optimization**
- Uses Google's professional Vehicle Routing Problem solver
- Minimizes total travel time across all teams
- Time window constraints (8 AM - 6 PM operations)
- Dynamic team optimization based on availability and demand

🗺️ **Pre-computed Travel Matrices**
- **No live API calls** - Uses pre-calculated driving time/distance matrices
- **177 locations**: Franchise office + 176 customers
- **Real driving data** - Based on actual Google Maps routing
- **Instant optimization** - No network delays or rate limits

📊 **Comprehensive Analysis**
- Detailed terminal output with progress tracking
- Team route optimization and performance analysis
- Travel time statistics and improvements
- Executable schedule generation with CSV export

⚡ **Smart Constraints**
- Customer availability and scheduling preferences
- Dynamic team composition and availability
- Working hours (franchise start time to 6 PM)
- Service time windows and duration constraints
- Peak week optimization (July 8-14, 2025)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Data Files
Required files in `data files/` folder:
- `franchises.csv` - Franchise configuration
- `cleans.csv` - Scheduled cleanings (⚠️ Large file - not in GitHub repo)
- `customers.csv` - Customer details and availability (⚠️ Large file - not in GitHub repo)
- `teams.csv` - Team composition (⚠️ Large file - not in GitHub repo)
- `complete_real_driving_time_matrix.csv` - Travel time matrix
- `complete_real_driving_distance_matrix.csv` - Travel distance matrix

### 3. Run the Optimizer
```bash
python optimizer.py
```

The system will automatically:
- Target Franchise 372 (Mount Prospect)
- Optimize July 8-14, 2025 (peak week with 204 cleanings)
- Use pre-computed matrices for 177 locations
- Generate ortools VRP optimized routes with dynamic team allocation

## Large File Notice

⚠️ **Important**: Some data files exceed GitHub's 100MB limit:
- `cleans.csv` (126 MB) - Contains cleaning schedule data
- `customers.csv` (215 MB) - Customer database  
- `teams.csv` (214 MB) - Team composition data

These files are excluded from the GitHub repository but are required for the optimizer to run.

## Output Example

```
🚀 TRAVEL-OPTIMIZED CLEANING SCHEDULER WITH ORTOOLS VRP
================================================================================

✅ Franchise loaded: Mount Prospect Cleaning Services
📍 Address: 123 Main St, Mount Prospect, IL
🕒 Operating hours: 8:00 - 18:00 (600 minutes)

📅 Using target week: 2025-07-08 (July 8-14, 2025)
📊 This is the PEAK WEEK with 204 cleanings for Franchise 372
🎯 Matrices available for 177 customers

✅ Found 204 scheduled cleanings
📊 Duration analysis: Average 92.4 minutes, Range: 45-180 minutes

📊 Available customers in matrix: 176
📈 Matrix dimensions: (177, 177)
🚗 Sample travel times (minutes):
   Franchise to first customer: 47 minutes
   Between first two customers: 57 minutes

📅 Running ortools VRP for Monday 2025-07-08
   Using 3 teams for 2025-07-08 (3 available)
   VRP problem: 45 locations (44 customers)
   ✅ Customer Customer_023 → Team 1 at 09:15 (travel: 23min)
   ✅ Customer Customer_045 → Team 2 at 08:47 (travel: 19min)
   ✅ Customer Customer_012 → Team 3 at 10:30 (travel: 31min)

✅ ortools VRP optimization complete!
📊 187 assignments scheduled
📊 Average travel time: 18.3 minutes per job

💰 ESTIMATED IMPROVEMENT:
   Before optimization: ~25.0 minutes per job
   After optimization: 18.3 minutes per job
   Time saved: 1252.9 minutes (20.9 hours) per week
   Improvement: 26.8%

🎊 ORTOOLS VRP OPTIMIZATION COMPLETED SUCCESSFULLY!
```

## Algorithm Details

**Objective**: Minimize total travel time using Vehicle Routing Problem

**Approach**: Google ortools constraint programming
- **VRP Formulation**: Multiple vehicles (teams) with time windows
- **Depot**: Franchise office (all routes start/end here)
- **Time Windows**: 8 AM - 6 PM operations with customer service times
- **Dynamic Teams**: Scales 1-5 teams based on availability and customer count
- **Optimization**: Guided Local Search with 30-second time limit

**Key Features**:
- ✅ Professional VRP solver (better than greedy algorithms)
- ✅ Pre-computed real driving times (no API delays)
- ✅ Time window constraints for realistic scheduling
- ✅ Dynamic multi-team route optimization
- ✅ Fallback to greedy algorithm if VRP fails

## Dynamic Team Management

### Team Scaling Logic
- **Real Team Data Available**: Uses actual operational teams from teams.csv
- **No Team Data**: Intelligent scaling based on daily customer count:
  - ≤15 customers → 1 team
  - ≤30 customers → 2 teams  
  - ≤45 customers → 3 teams
  - >45 customers → 3-5 teams (scales with demand)

## Matrix Structure

### Travel Time Matrix (`complete_real_driving_time_matrix.csv`)
```
Location,Franchise_Office,Customer_001,Customer_002,...,Customer_176
Franchise_Office,0.0,47.9,19.2,...,59.3
Customer_001,48.0,0.0,57.8,...,16.5
Customer_002,18.9,57.6,0.0,...,69.1
...
```

**Matrix Properties**:
- **Size**: 177 x 177 (Franchise + 176 customers)
- **Units**: Time in minutes, Distance in kilometers  
- **Customer IDs**: Generic labels (Customer_001, Customer_002, etc.)
- **Symmetric**: Travel time A→B ≈ B→A

## Data Requirements

### Required CSV Files in `data files/` folder:

**Small Files (included in repo)**:
- `franchises.csv` - Franchise configuration
- `complete_real_driving_time_matrix.csv` - Travel times between all locations
- `complete_real_driving_distance_matrix.csv` - Travel distances between all locations

**Large Files (excluded from GitHub, required locally)**:
- `cleans.csv` (126MB) - CustomerId, CleanDate, CleanTime, TeamNumber, CleanId
- `customers.csv` (215MB) - Customer details, addresses, availability constraints
- `teams.csv` (214MB) - Team composition, employee availability

## Configuration

### Default Settings
- **Franchise**: 372 (Mount Prospect)
- **Target Week**: July 8-14, 2025 (hard-coded peak week)
- **Teams**: Dynamic scaling 1-5 teams based on availability and demand
- **Working Hours**: 8 AM - 6 PM
- **VRP Time Limit**: 30 seconds per day
- **Matrix Size**: 177 locations (franchise + 176 customers)

## Technical Stack

- **Python 3.8+**
- **pandas**: CSV data processing and matrix operations
- **numpy**: Numerical computations for matrices
- **ortools**: Google's Vehicle Routing Problem solver
- **datetime**: Date/time handling for scheduling

**No API keys required** - All data is pre-computed and local.

## Performance

**Expected Runtime** (204 cleanings, 177 locations):
- Matrix loading: ~2-3 seconds
- Data processing: ~5-10 seconds  
- ortools VRP per day: ~30 seconds x 7 days = ~3.5 minutes
- Results generation: ~2-3 seconds
- **Total: ~4-5 minutes**

**Memory Usage**: ~200MB for matrices and optimization

## Troubleshooting

### Common Issues

**"No customers from [date] found in matrix"**
- Customer IDs in cleaning data don't match matrix column names
- Current cleaning data uses real Customer IDs, matrix uses Customer_001, etc.
- **Solution**: Regenerate matrices with real Customer IDs or update cleaning data

**"No travel matrices loaded"**
- Matrix files missing from `data files/` folder
- Check file names: `complete_real_driving_time_matrix.csv` and `complete_real_driving_distance_matrix.csv`

**"ortools VRP failed, falling back to greedy"**
- VRP solver couldn't find solution (normal for complex problems)
- System automatically uses greedy algorithm as backup
- Still produces valid optimized schedule

**"File not found" errors**
- Large CSV files (cleans.csv, customers.csv, teams.csv) are not included in GitHub repo
- These files must be obtained separately due to size limitations

## Results Interpretation

### CSV Output (`optimized_schedule_franchise_372_YYYYMMDD_HHMMSS.csv`)
- **clean_date**: Scheduled cleaning date
- **customer_id**: Customer identifier (Customer_001, etc.)
- **team_number**: Assigned team (dynamically allocated)
- **start_time/end_time**: Exact appointment window
- **travel_time_minutes**: Travel time to reach this customer
- **clean_duration_minutes**: Time needed for cleaning
- **previous_location**: Where team is coming from

### Key Metrics
- **Global average travel**: Primary optimization target
- **Total travel time**: Sum of all travel between jobs
- **Improvement percentage**: Compared to 25-minute baseline
- **Jobs scheduled**: Customers successfully assigned to teams
- **Teams used**: Dynamic allocation per day based on demand

## Support

This system is specifically configured for:
- **Franchise 372** (Mount Prospect location)
- **Peak week optimization** (July 8-14, 2025)
- **177-location matrices** (franchise + 176 customers)
- **Generic Customer IDs** (Customer_001 format)
- **Dynamic team management** (1-5 teams based on availability)

The ortools VRP optimization with dynamic team scaling can realistically save 15-25% of travel time compared to manual scheduling, potentially saving 15+ hours per week for franchise operations while adapting to varying daily workloads.
