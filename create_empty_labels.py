import os

img_dir = "C:/Users/Asus/OneDrive/Desktop/Hackathon/negative_samples_backgrounds"

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
            
            # Create an empty text file
            open(label_path, "w").close()

print("✅ Empty labels created for all negative background images")
