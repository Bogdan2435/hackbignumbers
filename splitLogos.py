# from PIL import Image
# import os

# def split_image(image_path, output_dir, index):
#     # Open the image file
#     img = Image.open(image_path)
#     # Calculate dimensions of each quarter
#     width, height = img.size
#     quarter_width = width // 2
#     quarter_height = height // 2
#     # Split the image
#     img1 = img.crop((0, 0, quarter_width, quarter_height))
#     img2 = img.crop((quarter_width, 0, width, quarter_height))
#     img3 = img.crop((0, quarter_height, quarter_width, height))
#     img4 = img.crop((quarter_width, quarter_height, width, height))
#     # Save the images
#     img1.save(os.path.join(output_dir, 'img' + str(index) +'.png'))
#     index += 1
#     img2.save(os.path.join(output_dir, 'img' + str(index) + '.png'))
#     index +=1
#     img3.save(os.path.join(output_dir, 'img' + str(index) + '.png'))
#     index +=1
#     img4.save(os.path.join(output_dir, 'img' + str(index) + '.png'))
#     index +=1


# # Example usage:
# index = split_image('logos-a1aautocenter.com/logo1.png', 'splited/', index = 1)


from PIL import Image
import os

def split_image(image_path, output_dir):
    # Open the image file
    img = Image.open(image_path)
    # Calculate dimensions of each quarter
    width, height = img.size
    quarter_width = width // 2
    quarter_height = height // 2
    # Split the image
    img1 = img.crop((0, 0, quarter_width, quarter_height))
    img2 = img.crop((quarter_width, 0, width, quarter_height))
    img3 = img.crop((0, quarter_height, quarter_width, height))
    img4 = img.crop((quarter_width, quarter_height, width, height))
    # Save the images
    img1.save(os.path.join(output_dir, 'img1-14.png'))
    img2.save(os.path.join(output_dir, 'img2-14.png'))
    img3.save(os.path.join(output_dir, 'img3-14.png'))
    img4.save(os.path.join(output_dir, 'img4-14.png'))

# Example usage:
split_image('logos-ocaseys.nl/logo3.png', 'splitted/')

