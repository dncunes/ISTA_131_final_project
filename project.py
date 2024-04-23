# 
# Author: Danielle N. Cunes
# Date: 4/22/2024
# Course: ISTA 131
# Assignment: Final Project
# File: project.py
# Purpose: The purpose of this program is to read in data from a data table found online, clean up the data and create three different figures with the data. 
#          This program uses data from female engineering students enrolled in a graduate program and compares them to the male engineering graduate students. 
# 

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def read_in_csv():
    '''
    This function uses the read_csv function to read in a csv and create a dataframe using the csv file.
    
    Parameters: 
        None

    Returns:
        grad_students_df: a pandas DataFrame
    '''
    column_names = ['Year', 'Total Grad Students', 'Male Grad Students', 'Male Grad Percent', 'Female Grad Students', 'Female Grad Percent', 
                    'Total Post-Doc Students', 'Male Post-Doc Students', 'Male Post-Doc Percent', 'Female Post-Doc Students', 'Female Post-Doc Percent', 
                    'Total Doctorate Students', 'Male Doctorate Students', 'Male Doctorate Percent', 'Female Doctorate Students', 'Female Doctorate Percent']
    
    grad_students_df = pd.read_csv("grad_students.csv", skiprows=[0,1,2,3,4,5,30,31,34,35,36,37,38,39,40,41,42,43,44,45,48,49,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69], names=column_names,
                                   thousands=',')

    return grad_students_df
    
def clean_up_data(grad_students_df):
    '''
    This function takes the DataFrame created in the previous function, cleans up the data by filling in missing data and extracting the data 
    needed to plot scatterplots, line plots, and bar charts.
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            grad_students_df: a pandas DataFrame
    '''
    # Calculate first average missing
    row_one = grad_students_df.iloc[0, 2]
    row_two = grad_students_df.iloc[2, 2]
    average1 = (row_one + row_two) / 2
    grad_students_df.iloc[1, 2] = average1

    # Calculate second value missing
    row_three = grad_students_df.iloc[0, 4]
    row_four= grad_students_df.iloc[2, 4]
    average2 = (row_three + row_four) / 2
    grad_students_df.iloc[1, 4] = average2

    return grad_students_df

def female_grad_scatterplot(grad_students_df):
    '''
    This function takes the DataFrame created in the read_csv function and uses the female graduate data to create a scatterplot with a line regression visual. 
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            None
    '''
    # Creating variables to call them when plotting
    x = grad_students_df['Year']
    y = grad_students_df['Female Grad Students']

    # Creating scatterplot, changing color, tick labels, axes
    plt.scatter(x,y)
    plt.xlabel('Year', color='darkmagenta', fontweight='bold')
    plt.tick_params(axis='x', colors='darkmagenta')
    plt.ylabel('Female Grad Students', color='darkmagenta', fontweight='bold')
    plt.tick_params(axis='y', colors='darkmagenta')
    plt.gca().spines['top'].set_color('darkmagenta')
    plt.gca().spines['bottom'].set_color('darkmagenta')
    plt.gca().spines['left'].set_color('darkmagenta')
    plt.gca().spines['right'].set_color('darkmagenta')
    plt.gca().set_facecolor('lavenderblush')
    plt.title('Female Grad Students with Regression Line', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'darkmagenta'})
    plt.scatter(grad_students_df['Year'], grad_students_df['Female Grad Students'], label='Female Grad Students', c='hotpink')
    plt.legend()

    #Creating line regression
    coefficients = np.polyfit(x,y,1)
    poly = np.poly1d(coefficients)
    plt.plot(x, poly(x), color='magenta')

def male_grad_scatterplot(grad_students_df):
    '''
    This function takes the DataFrame created in the read_csv function and uses the male graduate data to create a scatterplot with a line regression visual. 
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            None
    '''
    # Creating variables to call them when plotting
    x = grad_students_df['Year']
    y = grad_students_df['Male Grad Students']

    # Creating scatterplot, changing color, tick labels, axes
    plt.scatter(x,y)
    plt.xlabel('Year', color='navy', fontweight='bold')
    plt.tick_params(axis='x', colors='navy')
    plt.ylabel('Number of Grad Students', color='navy', fontweight='bold')
    plt.tick_params(axis='y', colors='navy')
    plt.gca().spines['top'].set_color('navy')
    plt.gca().spines['bottom'].set_color('navy')
    plt.gca().spines['left'].set_color('navy')
    plt.gca().spines['right'].set_color('navy')
    plt.gca().set_facecolor('aliceblue')
    plt.title('Male vs Female Grad Students', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'navy'})
    plt.scatter(grad_students_df['Year'], grad_students_df['Male Grad Students'], label='Male Grad Students', c='blue')
    plt.legend()

    # Creating line regression
    coefficients = np.polyfit(x,y,1)
    poly = np.poly1d(coefficients)
    plt.plot(x, poly(x), color='navy')

def create_line_plot(grad_students_df):
    '''
    This function takes the DataFrame created in the read_csv function and uses the female and male graduate data to create a line plot visual. 
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            None
    '''
    # Create variables to call them when plotting
    x = grad_students_df['Year']
    y1 = grad_students_df['Female Grad Students']
    y2 = grad_students_df['Male Grad Students']

    #Plot y1
    plt.plot(x, y1, label='Female Grad Students', color='deeppink')

    # Plot y2
    plt.plot(x, y2, label='Male Grad Students', color='blue')

    # Create the rest of the line plot
    plt.xlabel('Year', color='indigo', fontweight='bold')
    plt.ylabel('Number of Grad Students', color='indigo', fontweight='bold')
    plt.title('Male vs Female Grad Students', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'indigo'})
    plt.tick_params(axis='x', colors='indigo')
    plt.tick_params(axis='y', colors='indigo')
    plt.gca().spines['top'].set_color('indigo')
    plt.gca().spines['bottom'].set_color('indigo')
    plt.gca().spines['left'].set_color('indigo')
    plt.gca().spines['right'].set_color('indigo')
    plt.gca().set_facecolor('seashell')
    plt.legend()

def create_bar_chart(grad_students_df):
    '''
    This function takes the DataFrame created in the read_csv function and uses the female and male graduate data to create a bar chart visual. 
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            None
    '''
    # Create variables to call them when plotting
    x = grad_students_df['Year'].astype(str)
    y1 = grad_students_df['Female Grad Students']
    y2 = grad_students_df['Male Grad Students']

    # Set bar width
    bar_width = .3

    # Set the x indices
    x_indices = np.arange(len(x))

    #Plot y1 bars
    plt.bar(x_indices, y1, width=bar_width, label='Female Grad Students', color='hotpink', edgecolor='darkmagenta')

    #Plot y2 bars
    plt.bar(x_indices + bar_width, y2, width=bar_width, label='Male Grad Students', color='royalblue', edgecolor='navy')

    # Plot rest of chart
    plt.xlabel('Female Grad Students', color='indigo', fontweight='bold')
    plt.ylabel('Male Grad Students', color='indigo', fontweight='bold')
    plt.tick_params(axis='x', colors='indigo')
    plt.tick_params(axis='y', colors='indigo')
    plt.gca().spines['top'].set_color('indigo')
    plt.gca().spines['bottom'].set_color('indigo')
    plt.gca().spines['left'].set_color('indigo')
    plt.gca().spines['right'].set_color('indigo')
    plt.gca().set_facecolor('seashell')
    plt.title('Male vs Female Grad Students', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'indigo'})

    # Plot tick marks for bar chart
    plt.xticks(x_indices + bar_width / 2, x)
    plt.legend()
    plt.tight_layout()

def main():
    new_df = read_in_csv()
    clean_up_data(new_df)
    female_grad_scatterplot(new_df)
    male_grad_scatterplot(new_df)
    plt.figure(facecolor='lavenderblush')
    create_line_plot(new_df)
    plt.figure()
    create_bar_chart(new_df)
    plt.show()

if __name__ == "__main__":
    main()