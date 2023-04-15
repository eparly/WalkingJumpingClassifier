import os
import tkinter as tk
import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from connectFuncs import guiFile
from visualizations import plotPredictions


class PlotterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Acceleration Plotter and Classifier")
        self.open_button = ctk.CTkButton(
            master, text="Open File", command=self.select_file)
        self.open_button.pack(side="top")

        self.quit_button = ctk.CTkButton(master, text="Quit",
                                         command=self.master.destroy)
        self.quit_button.pack(side="bottom")

        self.welcome_label = ctk.CTkLabel(master, text='Welcome to the Acceleration Data Plotter!\n\n'
                                      'Please click the button below to select a CSV file\n'
                                      'containing raw acceleration data.')
        self.welcome_label.pack(pady=20)

        # create a canvas to display the plot
        self.canvas = ctk.CTkCanvas(master, width=1000, height=600)
        self.canvas.pack()

        self.filename = None
        self.num_points = None
        self.status_text = ctk.StringVar(value="No file selected.")
        self.status_label = ctk.CTkLabel(
            master, textvariable=self.status_text, relief=ctk.SUNKEN, anchor=ctk.W)
        self.status_label.pack(side=ctk.LEFT, fill=ctk.X)

        self.predictions = None
        self.prediction_text = ctk.StringVar(value="No predictions yet.")
        self.prediction_label = ctk.CTkLabel(
            master, textvariable=self.prediction_text, relief=ctk.SUNKEN, anchor=ctk.W)
        self.prediction_label.pack(side=ctk.RIGHT, fill=ctk.X)

    def select_file(self):

        # open a file dialog to select a csv file
        filename = askopenfilename(filetypes=[("CSV Files", "*.csv")])
        df = pd.read_csv(filename)
        self.filename = os.path.basename(filename)
        self.num_points = len(df)
        if(self.filename):
            self.status_text.set(
                "File: {} | Number of Points: {}".format(self.filename, self.num_points))
        else:
            self.status_text.set("No file selected.")



        fig = plotPredictions(filename)
        # clear the existing plot and update the canvas with the new plot
        self.welcome_label.destroy()
        self.canvas.delete()
        self.plot_widget = FigureCanvasTkAgg(fig, master=self.canvas)
        self.plot_widget.draw()

        # resize the canvas to fit the plot
        self.plot_widget.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

        _, predictions = guiFile(filename)
        self.prediction_text.set(predictions)


if __name__ == '__main__':
    root = ctk.CTk()
    app = PlotterGUI(root)
    root.mainloop()
