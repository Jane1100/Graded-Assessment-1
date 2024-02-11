import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#PLEASE DOWNLOAD THE DATASET FROM KAGGLE AND REPLACE IT BELOW WITH THE PATH ON YOUR PC. THE NAME OF THE DATASET ON KAGGLE IS "Top 200 Movies of 2023 Dataset"
#AND THE CSV FILE IS CALLED "Top_200_Movies_Dataset_2023(Cleaned).csv"
#It did not work with me to import it directly from the website, sorry :(. But all the code is working correctly.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

dataset = pd.read_csv('C:\\Users\\Jane Abi Saad\\Downloads\\archive (8)\\Top_200_Movies_Dataset_2023(Cleaned).csv', encoding='latin1')
length = len(dataset)
print("Length of dataset: ", length)

#Distribution of Release Dates (how many movies were released in this date)
plt.figure(figsize=(10, 6))
plt.hist(pd.to_datetime(dataset['Release Date']), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Release Dates')
plt.xlabel('Release Date')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


dataset['Total Gross'] = dataset['Total Gross'].str.replace(',', '').str.replace('$', '').astype(float)

dataset['Rank'] = pd.to_numeric(dataset['Rank'], errors='coerce')

dataset.dropna(subset=['Rank', 'Total Gross'], inplace=True)

x = dataset['Rank']
y = dataset['Total Gross']

#Correlation between rank and total gross of movie
corr_coeff = np.corrcoef(x, y)[0, 1]

print("Correlation coefficient between Rank and Total Gross:", corr_coeff)

#Linear Distribution between Rank and Box Office Earnings
if x.empty or y.empty:
    print("Error: One or both of the variables are empty.")
else:
    slope, intercept = np.polyfit(x, y, 1)

    predicted_y = slope * x + intercept
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='skyblue', label='Data Points')
    
    plt.plot(x, predicted_y, color='red', label='Linear Regression')
    
    plt.title('Linear Distribution between Rank and Box Office Earnings')
    plt.xlabel('Rank')
    plt.ylabel('Box Office Earnings')
 
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()


dataset['Theaters'] = dataset['Theaters'].str.replace(",", "").str.replace("'-","").str.replace("-","")

dataset = dataset[dataset['Theaters'] != '']
dataset = dataset[dataset['Total Gross'] != '']

dataset['Theaters'] = dataset['Theaters'].astype(float)
dataset['Total Gross'] = dataset['Total Gross'].astype(float)

dataset.dropna(subset=['Theaters', 'Total Gross'], inplace=True)
z = dataset['Total Gross']
u = dataset['Theaters']

#Correlation between Box Office Earnings and Number of Theaters
corr_coeff = np.corrcoef(z, u)[0, 1]

print("Correlation coefficient between Box Office Earnings and Number of Theaters:", corr_coeff)

#Linear Distribution between Box Office Earnings and Number of Theaters
if z.empty or u.empty:
    print("Error: One or both of the variables are empty.")
else:
    slope, intercept = np.polyfit(z, u, 1)

    predicted_y = slope * z + intercept
    
    plt.figure(figsize=(10, 6))
    plt.scatter(z, u, color='skyblue', label='Data Points')
    
    plt.plot(z, predicted_y, color='red', label='Linear Regression')
    
    plt.title('Linear Distribution between Box Office Earnings and Number of Theaters')
    plt.xlabel('Box Office Earnings')
    plt.ylabel('Number of Theaters')
 
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()



