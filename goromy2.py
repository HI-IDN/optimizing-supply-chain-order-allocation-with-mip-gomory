

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data source
mainFile = pd.ExcelFile("Supply chain logisitcs problem.xlsx")

# reading in all from excel sheets
# using dictionary to access all dataframe varaible easier
df_dict = {}

for names in mainFile.sheet_names:
    globals()[names] = mainFile.parse(names)
    df_dict[names] = globals()[names]
# show all keys    
df_dict.keys()

# Load data source
mainFile = pd.ExcelFile("Supply chain logisitcs problem.xlsx")

# reading in all from excel sheets
# using dictionary to access all dataframe varaible easier
df_dict = {}

for names in mainFile.sheet_names:
    globals()[names] = mainFile.parse(names)
    df_dict[names] = globals()[names]
# show all keys    
df_dict.keys()

for df_name, df in df_dict.items():
    print(df_name, '- shape:', df.shape)
    duplicate_count = df.duplicated().sum()
    missing_values_count = df.isnull().sum().sum()
    
    if duplicate_count > 0 or missing_values_count > 0:
        print(f">>>>{df_name} - duplicates: {duplicate_count}; missing values: {missing_values_count}")

# drop duplicates
FreightRates = FreightRates.drop_duplicates()
# update dictionary
df_dict['FreightRates'] = FreightRates

for df_name, df in df_dict.items():
    df.columns = [col.strip().replace(' ', '_').replace('/', '_').upper() for col in df.columns]

orderList = df_dict['OrderList']  # Get the OrderList dataframe

# Merge OrderList with FreightRates and WhCosts on relevant columns
orderList = orderList.merge(df_dict['FreightRates'], left_on=['CARRIER', 'ORIGIN_PORT', 'DESTINATION_PORT'],
                            right_on=['CARRIER', 'ORIG_PORT_CD', 'DEST_PORT_CD'], how='left')
orderList = orderList.merge(df_dict['WhCosts'], left_on='PLANT_CODE', right_on='WH', how='left')

# Calculate the cost by multiplying unit quantity with the shipping rate and adding the storage cost
orderList['COST'] = (orderList['UNIT_QUANTITY'] * orderList['RATE']) + (orderList['UNIT_QUANTITY'] * orderList['COST_UNIT'])

# Perform detailed data preparation and transformation for OrderList dataframe
orderList = orderList.dropna()  # Drop rows with missing values
orderList['ORDER_DATE'] = pd.to_datetime(orderList['ORDER_DATE'])

# Perform additional data preparation and transformation for other dataframes
# Assuming you want to perform similar steps for ProductsPerPlant dataframe
productsPerPlant = df_dict['ProductsPerPlant']
productsPerPlant = productsPerPlant.dropna()  # Drop rows with missing values

# update dictionary values
df_dict['OrderList'] = orderList
df_dict['ProductsPerPlant'] = productsPerPlant

# Print the updated OrderList dataframe
orderList

for df_name, df in df_dict.items():
    print(f'Dataframe: {df_name} >>>')
    print('Columns:', df.columns.values)
    
df_dict['OrderList'].describe()
df_dict['FreightRates'].describe()

correlation_matrix = df_dict['OrderList'].corr(numeric_only=True).round(2)
correlation_matrix

# plot correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

pricing_strategy = df_dict['FreightRates']['MODE_DSC'].unique()
price_elasticity = df_dict['FreightRates']['RATE'].mean()
historical_revenue = df_dict['OrderList']['UNIT_QUANTITY'] * df_dict['OrderList']['COST']
total_revenue = historical_revenue.sum()
print("Pricing strategies:", pricing_strategy)
print("Average price elasticity:", price_elasticity)
print("Total historical revenue:", total_revenue)

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar plot of service levels (left subplot)
sns.countplot(data=df_dict['OrderList'], x='SERVICE_LEVEL', ax=axes[0])
axes[0].set_title('Distribution of Service Levels')
axes[0].set_xlabel('Service Level')
axes[0].set_ylabel('Count')

# Box plot of daily capacities (right subplot)
sns.boxplot(data=df_dict['WhCapacities'], y='DAILY_CAPACITY', ax=axes[1])
axes[1].set_title('Distribution of Daily Capacities')
axes[1].set_ylabel('Daily Capacity')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# Scatter plot of weight vs. shipping cost
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_dict['OrderList'], x='WEIGHT', y='COST', hue='CARRIER')
plt.title('Weight vs. Shipping Cost')
plt.xlabel('Weight')
plt.ylabel('Cost')
plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Number of Products that each plant manufactures
plt.sca(axes[0])
plt.xticks(rotation=45)
plant_counts = pd.DataFrame(df_dict['ProductsPerPlant']['PLANT_CODE'].value_counts())
# ax.bar(plant_counts.index, plant_counts.to_numpy().reshape(-1))
plt.bar(plant_counts.index, plant_counts.to_numpy().reshape(-1))
plt.title("Number of Products that each plant manufactures")

# Plot Manufacturing Cost for each Plant
plt.sca(axes[1])
plt.xticks(rotation=45)
axes[1].bar(df_dict['WhCosts']['WH'], df_dict['WhCosts']["COST_UNIT"])
axes[1].set_title("Manufacturing Cost for each Plant")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

import plotly.graph_objects as go

# Load the PlantPorts dataframe
df_plant_ports = df_dict['PlantPorts']

# Create the figure
fig = go.Figure()

# Add the connections as traces
for _, row in df_plant_ports.iterrows():
    fig.add_trace(
        go.Scatter(
            x = [row['PLANT_CODE'], row['PORT']],
            y = [1, 0],
            mode = 'lines+markers',
            marker = dict(
                size = 10,
                symbol = 'circle',
                line = dict(
                    color = 'blue',
                    width = 2
                )
            ),
            hoverinfo = 'text',
            text = f"Plant: {row['PLANT_CODE']}<br>Port: {row['PORT']}",
        )
    )

# Set up the layout
fig.update_layout(
    title_text = 'Plant and Port Connections',
    showlegend = False,
    xaxis = dict(
        title = 'PLANT_CODE - PORT',
        tickangle = -45
    ),
    yaxis = dict(
        title = '',
        showticklabels = False,
        range = [-0.2, 1.2]
    )
)

# Show the interactive plot
fig.show()

# Creates a list of all the supply nodes
supply_nodes = list(df_dict['WhCosts']['WH'])

# Creates a dictionary for the number of units of supply for each supply node
supply_dict = {}
for node in supply_nodes:
    total_capacity = sum(df_dict['WhCapacities']['DAILY_CAPACITY'][df_dict['WhCapacities']['PLANT_ID'] == node])
    supply_dict[node] = total_capacity

# Creates a list of all demand nodes
demand_nodes = list(df_dict['OrderList']['DESTINATION_PORT'].unique())

# Creates a dictionary for the number of units of demand for each demand node
demand_dict = {}
for index, row in df_dict['OrderList'].iterrows():
    dest_port = row['DESTINATION_PORT']
    unit_quantity = row['UNIT_QUANTITY']
    if dest_port in demand_dict:
        demand_dict[dest_port] += unit_quantity
    else:
        demand_dict[dest_port] = unit_quantity
        
costs = []
for index, row in df_dict['OrderList'].iterrows():
    carrier = row['CARRIER']
    orig_port = row['ORIG_PORT_CD']
    dest_port = row['DEST_PORT_CD']
    weight = row['WEIGHT']
    
    matching_rates = df_dict['FreightRates'][(df_dict['FreightRates']['CARRIER'] == carrier) &
                                  (df_dict['FreightRates']['ORIG_PORT_CD'] == orig_port) &
                                  (df_dict['FreightRates']['DEST_PORT_CD'] == dest_port) &
                                  (df_dict['FreightRates']['MINM_WGH_QTY'] <= weight) &
                                  (df_dict['FreightRates']['MAX_WGH_QTY'] >= weight)]['RATE']
    
    rate = matching_rates.values[0] if not matching_rates.empty else None
    costs.append(rate)
    
    cost_dict = {}
for supply_node in supply_nodes:
    cost_dict[supply_node] = {}
    for demand_node in demand_nodes:
        cost_dict[supply_node][demand_node] = costs.pop(0)
        
from pulp import *

# Creates the 'prob' variable to contain the problem data
prob = LpProblem("MaterialSupplyProblem", LpMinimize)

# Creates a list of tuples containing all the possible routes for transport
Routes = [(w, b) for w in supply_nodes for b in demand_nodes]

# A dictionary called 'Vars' is created to contain the referenced variables(the routes)
vars = LpVariable.dicts("Route", (supply_nodes, demand_nodes), 0, None, LpInteger)

# The minimum objective function is added to 'prob' first
prob += lpSum([vars[w][b] * cost_dict[w][b] for (w, b) in Routes]), "Sum_of_Transporting_Costs"

for w in supply_nodes:
    prob += (
        lpSum([vars[w][b] for b in demand_nodes]) <= supply_dict[w],
        "Sum_of_Products_out_of_warehouses_%s" % w,
    )

# The demand minimum constraints are added to prob for each demand node (project)
for b in demand_nodes:
    prob += (
        lpSum([vars[w][b] for w in supply_nodes]) >= demand_dict[b],
        "Sum_of_Products_into_projects%s" % b,
    )

# Add constraint to avoid routes from a node to itself
for w in supply_nodes:
    for b in demand_nodes:
        if w != b:  # Check if supply node is different from demand node
            prob += (vars[w][b] >= 0, "No_Self_Transportation_%s_%s" % (w, b))
        else:
            prob += (vars[w][b] == 0, "No_Self_Transportation_%s_%s" % (w, b))
            vars[w][b].lowBound = 0  # Set lower bound to 0 for self-transportation
            

prob.solve()

# Print the variables optimized value
for v in prob.variables():
    print(v.name, "=", v.varValue)
    
# The optimised objective function value is printed to the screen
print("Value of Objective Function = ", value(prob.objective))