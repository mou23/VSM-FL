import sys
import json

def calculate_accuracy_at_k(data):
    for top in [1, 5, 10, 15, 20, 25, 30]:
        count = 0
        total_bug = 0
        for key, value in data.items():
            # print(key)
            suspicious_files = value['results']
            # print(suspicious_files)
            fixed_files = value['truth']
            # print(fixed_files) 
            # print(suspicious_files[0:top])
            for fixed_file in fixed_files:
                if fixed_file in suspicious_files[0:top]:
                    print(key,fixed_file)
                    count = count + 1
                    break
            total_bug = total_bug + 1
        print('accuracy@', top, count, total_bug, (count*100/total_bug))

def calculate_mean_reciprocal_rank_at_k(data):
    for top in [10, 20, 30, 40, 50]:
        total_bug = 0
        inverse_rank = 0
        for key, value in data.items():
            suspicious_files = value['results']
            length_of_suspicious_files = len(suspicious_files)
            fixed_files = value['truth']
            minimum_length = min(top,length_of_suspicious_files)
            for i in range(minimum_length):
                if(suspicious_files[i] in fixed_files):
                    print('first rank', key, i+1, suspicious_files[i])
                    inverse_rank = inverse_rank + (1/(i+1))
                    break
            total_bug = total_bug + 1
        if inverse_rank == 0:
            print("MRR@", top, 0)
        else:
            print("MRR@", top, (1/total_bug)*inverse_rank)

def calculate_mean_average_precision_at_k(data):
    for top in [10, 20, 30, 40, 50]:
        total_bug = 0
        total_average_precision = 0
        for key, value in data.items():
            average_precision = 0
            precision = 0
            suspicious_files = value['results']
            length_of_suspicious_files = len(suspicious_files)
            fixed_files = value['truth']
            number_of_relevant_files = 0
            minimum_length = min(top,length_of_suspicious_files)
            for i in range(minimum_length):
                # print("i",i)
                if(suspicious_files[i] in fixed_files):
                    print(key,suspicious_files[i], " relevant")
                    number_of_relevant_files = number_of_relevant_files + 1                        
                    precision = precision + (number_of_relevant_files/(i+1))
                    # print("precision ", precision)
            average_precision = precision/len(fixed_files)
            # print("average_precision" ,average_precision, len(fixed_files))
            total_average_precision = total_average_precision + average_precision
            total_bug = total_bug + 1
        mean_average_precision = total_average_precision/total_bug
        print("MAP@", top, mean_average_precision)

project = sys.argv[1]     
with open(project+'/results.json', 'r') as file:
    data = json.load(file)

calculate_accuracy_at_k(data)
calculate_mean_reciprocal_rank_at_k(data)
calculate_mean_average_precision_at_k(data)