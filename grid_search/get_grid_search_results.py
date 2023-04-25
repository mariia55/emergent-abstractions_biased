import os
import re
import pandas as pd

# results should be stored in a pandas dataframe and exported as csv
data = pd.DataFrame()

# go through all files in grid search results folder
directory = "test"
for filename in os.listdir(directory):
    if filename.endswith(".out"):
        with open(os.path.join(directory, filename), "r") as file:
            lines = file.readlines()
            header = lines[0].strip()

            # create a dictionary with parameters
            parameters = {}
            matches = re.findall(r"--(.+?)=(.+?)\"", header)
            for match in matches:
                parameters[match[0]] = match[1]

            #print(parameters)

            # get results
            if len(lines) >= 3:
                final_train = lines[-3].strip()
                final_test = lines[-2].strip()

            train_loss, train_acc = re.findall(r"\"loss\": (.+?), \"acc\": (.+?),", final_train)[0]
            test_loss, test_acc = re.findall(r"\"loss\": (.+?), \"acc\": (.+?),", final_test)[0]

            # update the parameters dictionary with train and test accuracies
            parameters['train_loss'] = train_loss
            parameters['train_accuracy'] = train_acc
            parameters['test_loss'] = test_loss
            parameters['test_accuracy'] = test_acc

            df = pd.DataFrame(parameters, index=[0])

            data = pd.concat([data, df], ignore_index=True)

# sort and save as csv
data = data.astype(float)
data[['attributes', 'values', 'game_size', 'batch_size', 'hidden_size']] = data[['attributes', 'values', 'game_size', 'batch_size', 'hidden_size']].astype(int)
data = data.sort_values(by=['attributes', 'values', 'game_size', 'batch_size', 'learning_rate', 'hidden_size', 'temperature', 'temp_update'])

data.to_csv('results.csv', index=False)