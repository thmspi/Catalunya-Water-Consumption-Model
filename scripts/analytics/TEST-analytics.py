# In[1]: 

import analytics_functions as af
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_dir, '..', '..','data', 'final_results', 'test-complete_census_all.csv')
result_path = os.path.abspath(result_path)
results = pd.read_csv(result_path) 

coordinates_path = os.path.join(script_dir, '..', '..','data', 'base_data', 'farms with coordinates.xlsx')
coordinates_path = os.path.abspath(coordinates_path)
coordinates = pd.read_excel(coordinates_path) 

# In[2]: 
choice = 'y'
while True:
    print('Here is what you can do :'
            '\n 1) Plot total water consumption for a specific farm'
            '\n 2) Analyze water consumption depending on the farm location'
            '\n 3) Compare fattening cycle and their water consumption'
            '\n 4) Calculate total water consumption of Catalonia over a time period'
            '\n 5) Exit ')
    function = input("Please select an option : ").strip().lower()
    if function == '1':
        farm_id = input("Enter a farm MO (0000AA) : ").strip()
        af.plot_farm_data(results, farm_id)
    elif function =='2':
        af.analyze_geographical_correlation(results, coordinates)
    elif function =='3':
        farm_id = input("Enter a farm MO (0000AA) : ").strip()
        af.compare_fattening_cycles(results, farm_id)
    elif function =='4':
        af.summarize_consumption_by_date(results)
    elif function =='5':
        print('Exiting ...')
        break
    else :
        print('Invalid option')
