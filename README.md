Airshed Advance Traffic Monitoring System (AATMS)
Welcome to the Airshed Advance Traffic Monitoring System (AATMS)! This project provides an advanced traffic monitoring solution designed for Intelligent Transportation Systems (ITS). The system allows users to track vehicles crossing multiple predefined lines in video footage, enabling route and time tracking for different vehicle classes.

This software combines state-of-the-art object detection and tracking algorithms with an intuitive Tkinter GUI, making it easy to use for both researchers and field engineers.

Features
Line Drawing on First Frame

Users can draw multiple line segments on the first frame of the video.
These lines act as checkpoints to monitor vehicle crossings.
Vehicle Tracking by Route

Tracks the time of crossing for each vehicle based on the bottom bounding box coordinates.
Allows users to analyze vehicle routes through different checkpoints.
Class-Specific Tracking

Differentiates vehicle classes (e.g., cars, bikes, buses) and logs their movements.
Detailed Outputs

Generates detailed reports with timestamps for vehicle crossings.
Saves results in CSV format for further analysis.
User-Friendly Tkinter GUI

Step-by-step guidance for loading videos, drawing lines, and running the analysis.
Real-time feedback on processing status.
Scalable for ITS Applications

Easily deployable at junctions and streets for traffic monitoring.
Installation
Requirements
Python 3.8 or later
Required libraries listed in requirements.txt
Steps
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/AATMS.git
cd AATMS
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Place your trained YOLO model file (e.g., .pt file) in the models directory.

Run the software:

bash
Copy code
python main.py
Usage Instructions
Step 1: Load a Video
Open the software and load a video file (supported format: .mp4).
Ensure the video shows the area you want to monitor.
Step 2: Draw Lines
The first frame of the video will be displayed.
Use the interactive tools to draw up to 4 lines.
These lines will act as checkpoints for monitoring.
Step 3: Start Analysis
Run the analysis, and the software will process the video frame by frame.
It detects and tracks vehicles crossing the predefined lines.
Step 4: Export Results
After processing, the software generates a CSV file with:
Vehicle class
Line crossed
Timestamp of crossing
Use this data for further insights and reporting.
Output Example
Here’s an example of the CSV file structure:

Vehicle ID	Class	Line Crossed	Crossing Time
1	Car	Line 1	2024-11-22 14:23:45
2	Bike	Line 2	2024-11-22 14:24:12
3	Bus	Line 1	2024-11-22 14:24:50
Applications
Traffic Flow Analysis: Monitor traffic at junctions and streets.
Route Optimization: Understand vehicle route patterns.
ITS Integration: Use data for real-time decision-making and smart city planning.
Contributing
We welcome contributions! If you’d like to contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a detailed explanation of your changes.
Support
For any issues or suggestions, feel free to raise an issue on the repository or contact us at support@airshed.com.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Start monitoring traffic smarter and faster with AATMS! 🚗🛵🚍
