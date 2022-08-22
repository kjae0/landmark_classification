from torchvision.transforms import functional

def rotation(image, degree):
    return functional.rotate(image, degree)

def cropping(image, top, left, height, width):
    return functional.crop(image, top, left, height, width)

def brighteness(image, degree):
    return functional.adjust_brightness(image, degree)

def erasing(image, top, left, height, width):
    return functional.erase(image, top, left, height, width)



