import os
import subprocess

def run_minimization(input_folder, output_folder, log_folder):
    # Ensure the output and log directories exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    # Get all the .sdf files in the input folder
    sdf_files = [f for f in os.listdir(input_folder) if f.endswith('.sdf')]
    
    for sdf_file in sdf_files:
        input_file = os.path.join(input_folder, sdf_file)
        
        # Define output file paths
        output_file = os.path.join(output_folder, f"minimized_{sdf_file}")
        log_file = os.path.join(log_folder, f"{sdf_file}_log.txt")
        
        # Open Babel command to run the minimization
        cmd = [
            'obabel', 
            '-isdf', input_file,
            '-O', output_file,
            '--minimize',
            '--ff', 'UFF',       # Force field (you can change to MMFF94 or others)
            '--steps', '20000',
            '--log', log_file
        ]
        
        try:
            # Run the Open Babel command
            subprocess.run(cmd, check=True)
            print(f"Minimization for {sdf_file} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {sdf_file}: {e}")
            continue

if __name__ == '__main__':
    # Input and output folder paths
    input_folder = '/home/bio-tech/Desktop/AC1/Project_7th_sem_moldec/missing_files/'
    output_folder = '/home/bio-tech/Desktop/AC1/Project_7th_sem_moldec/new_min/'
    log_folder = '/home/bio-tech/Desktop/AC1/Project_7th_sem_moldec/logs/'

    # Run the minimization process
    run_minimization(input_folder, output_folder, log_folder)