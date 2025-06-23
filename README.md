# smart_attendence

This project is a Smart Attendance System that uses facial recognition to mark attendance automatically. It includes image datasets, Python scripts for attendance processing, and necessary requirements files.

## Project Structure

- `AttendanceProject.py`: Main Python script for the attendance system.
- `ImagesAttendance/`: Directory containing images of known individuals.
- `UnknownFaces/`: Directory containing images of unknown faces detected.
- `att.csv`: CSV file presumably used for attendance records.
- `requirements.txt` and `requirements_no_tensorflow.txt`: Python dependencies for the project.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/Anixh06/smart_attendence.git
   cd smart_attendence
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
   or if you want to install without TensorFlow:
   ```
   pip install -r requirements_no_tensorflow.txt
   ```

3. Run the attendance system:
   ```
   python AttendanceProject.py
   ```

## Upload to GitHub Repository

The following steps were taken to upload the project to the GitHub repository:

1. Initialized a git repository in the project directory (if not already initialized).
2. Added the remote origin pointing to the GitHub repository URL: `https://github.com/Anixh06/smart_attendence`.
3. Added all project files to the git staging area.
4. Committed the files with the message: "Add existing project files to repository".
5. Pushed the commit to the `main` branch of the remote repository.

## Notes

- Make sure you have Python installed on your system.
- The project uses facial recognition libraries which may require additional system dependencies.
- For any issues or questions, please refer to the project repository or contact the maintainer.
