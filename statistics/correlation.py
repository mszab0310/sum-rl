import matplotlib.pyplot as plt
import pandas as pd


def calculate_correlation(file_path, waiting_columns, occupancy_columns):
    df = pd.read_excel(file_path)

    avg_speed = df[waiting_columns].mean(axis=1)
    avg_occupancy = df[occupancy_columns].mean(axis=1)

    correlation = avg_speed.corr(avg_occupancy)

    plt.scatter(avg_speed, avg_occupancy)
    plt.xlabel('Lane occupancy')
    plt.ylabel('Halting numbers')
    plt.title(f'Lane occupancy vs Halting numbers {round(correlation,2)}')

    plt.show()

    return correlation


file_path = "../observation_data.xlsx"
waiting_times = ['sum1', 'sum2', 'sum3', 'sum4', 'sum5', 'sum6', 'sum7', 'sum8']
occupancy_columns = ['occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6', 'occ7', 'occ8']
halting_number = ['veh_nr1', 'veh_nr2', 'veh_nr3', 'veh_nr4', 'veh_nr5', 'veh_nr6', 'veh_nr7', 'veh_nr8']

result = calculate_correlation(file_path, occupancy_columns, halting_number)
print("Correlation:", result)
