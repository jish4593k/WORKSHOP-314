# -*- coding: utf-8 -*-
# import libraries
import urllib.request as urllib2
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk

# specify the url
url = "http://www.accuweather.com/en/in/chennai/206671/current-weather/206671"

# get the web page content and store it in a variable 'webpage’
webpage = urllib2.urlopen(url)

# parse the html using beautiful soup and store in variable `soup`
soup = BeautifulSoup(webpage, "html.parser")

# finding the block we need
temp_block = soup.find('span', attrs={'class': 'large-temp'})
stats_block = soup.find('ul', attrs={'class': 'stats'})
sunrise_block = soup.find('ul', attrs={'class': 'time-period'})

# strip() is used to remove starting and trailing tags
temp = temp_block.text.strip()
stats = stats_block.text.strip()
sunrise = sunrise_block.text.strip()

print("Temperature:" + temp)
print("\nStats:\n\n" + stats)
print("\nSunrise:\n\n" + sunrise)

temp = temp.encode('utf-8')
stats = stats.encode('utf-8')
sunrise = sunrise.encode('utf-8')

# Save weather data to CSV
with open('weather.csv', 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([temp, stats, sunrise, datetime.now()])

# PyTorch Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# GUI setup
root = tk.Tk()
root.title("Weather Information")

# Display weather information
temp_label = tk.Label(root, text=f"Temperature: {temp}")
temp_label.pack()

stats_label = tk.Label(root, text=f"Stats: {stats}")
stats_label.pack()

sunrise_label = tk.Label(root, text=f"Sunrise: {sunrise}")
sunrise_label.pack()

# Neural Network setup
input_size = 3
hidden_size = 2
output_size = 1

model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('weather_model.pth'))

# Predict using the neural network
input_data = torch.Tensor([float(temp), float(stats), float(sunrise)])
prediction = model(input_data).item()

prediction_label = tk.Label(root, text=f"Neural Network Prediction: {prediction:.2f}")
prediction_label.pack()

root.mainloop()
