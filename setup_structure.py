import os
import shutil

def create_project_structure():
    """Create the complete project structure"""
    
    # Base directories
    directories = [
        # Data
        'data/raw/ml-100k',
        'data/processed',
        'data/external',
        
        # Source code
        'src/data',
        'src/models',
        'src/features',
        'src/similarity',
        'src/optimization',
        'src/evaluation',
        'src/utils',
        
        # Notebooks
        'notebooks',
        
        # Experiments
        'experiments',
        
        # Tests
        'tests',
        
        # Results
        'results/models',
        'results/predictions',
        'results/figures',
        'results/reports',
        
        # Configs
        'configs/experiment_configs',
        
        # Scripts
        'scripts',
    ]
    
    print("Creating folder structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/features/__init__.py',
        'src/similarity/__init__.py',
        'src/optimization/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py',
    ]
    
    print("\nCreating __init__.py files...")
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Module initialization"""\n')
        print(f"✓ Created: {init_file}")
    
    # Move your existing ml-100k data
    source_data = r"C:\Users\aravi\OneDrive\Desktop\ml-100k"
    target_data = "data/raw/ml-100k"
    
    print(f"\n📦 Copying MovieLens data...")
    print(f"   From: {source_data}")
    print(f"   To: {target_data}")
    
    # Copy all files from ml-100k
    if os.path.exists(source_data):
        for file in os.listdir(source_data):
            src_file = os.path.join(source_data, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, target_data)
        print("✓ Data copied successfully")
    else:
        print("⚠ Source data not found, please copy manually")
    
    # Move existing processed files
    processed_files = [
        'processed_ratings.csv',
        'processed_users.csv',
        'processed_movies.csv',
        'rating_matrix.csv'
    ]
    
    print("\n📦 Moving processed files...")
    for file in processed_files:
        if os.path.exists(file):
            shutil.move(file, f'data/processed/{file}')
            print(f"✓ Moved: {file}")
    
    print("\n" + "="*60)
    print("✅ PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*60)
    
if __name__ == "__main__":
    create_project_structure()
