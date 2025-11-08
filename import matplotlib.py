import matplotlib.pyplot as plt

# Data
labels = ['Training Samples (9,920)', 'Test Samples (2,479)']
sizes = [9920, 2479]

# Create pie chart (no specific colors set — requirement)
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Training vs Test Dataset Split (Total: 12,399 samples)')
plt.show()
