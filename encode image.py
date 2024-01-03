import base64
import io
import matplotlib.pyplot as plt
from PIL import Image

encoded_image = ""

# Decode base64 image data

image_data = base64.b64decode(encoded_image)

# Create a PIL Image object
pil_image = Image.open(io.BytesIO(image_data))

# Display the image using matplotlib
plt.imshow(pil_image)
plt.axis('off')
plt.show()
