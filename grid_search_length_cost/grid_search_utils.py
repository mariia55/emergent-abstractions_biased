import os
import re
import pandas as pd
    
def get_grid_search_results(directory, n_lines=125, header_in_line=0):

    # results should be stored in a pandas dataframe and exported as csv
    data = pd.DataFrame()

    # go through all files in grid search results folder
    directory = directory
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".out"):
            with open(os.path.join(directory, filename), "r") as file:
                lines = file.readlines()
                
                if len(lines) == n_lines: # if files have less lines, the search did not run for 100 epochs
                    header = lines[header_in_line].strip()

                    # create a dictionary with parameters
                    parameters = {}
                    matches = re.findall(r"--(.+?)=(.+?)\"", header)
                    for match in matches:
                        parameters[match[0]] = match[1]

                    #print(parameters)

                    # get results
                    final_train = lines[-3].strip()
                    final_test = lines[-2].strip()

                    train_loss, train_acc, train_length = re.findall(r"\"loss\": (.+?), \"acc\": (.+?), \"length\": (.+?),", final_train)[0]
                    test_loss, test_acc, test_length = re.findall(r"\"loss\": (.+?), \"acc\": (.+?), \"length\": (.+?),", final_test)[0]

                    # update the parameters dictionary with train and test accuracies
                    parameters['train_loss'] = train_loss
                    parameters['train_accuracy'] = train_acc
                    parameters['train_length'] = train_length
                    parameters['test_loss'] = test_loss
                    parameters['test_accuracy'] = test_acc
                    parameters['test_length'] = test_length

                    df = pd.DataFrame(parameters, index=[0])

                    data = pd.concat([data, df], ignore_index=True)

    if 'context_unaware' in data:
        data['context_unaware'] = data['context_unaware'].map({'True': 1.0, 'False': 0.0})
    # sort and save as csv
    data = data.astype(float)
    print(data)
    data[['attributes', 'values', 'game_size', 'batch_size', 'hidden_size']] = data[['attributes', 'values', 'game_size', 'batch_size', 'hidden_size']].astype(int)
    data = data.sort_values(by=['attributes', 'values', 'game_size', 'batch_size', 'learning_rate', 'hidden_size', 'temperature', 'temp_update'])

    data.to_csv('results_' + directory + '.csv', index=False)