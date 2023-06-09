import os
import fitz



pdfs_path = "/path/to/doc_pdfs"
img_path = "/path/to/imgs"

for fn in os.listdir(pdfs_path):
    # print(fn)
    name = fn[:-4]
    # print(name)

    pdf = fitz.open(os.path.join(pdfs_path, fn))

    for i,p in enumerate(pdf): 
    # page = pdf[pg]
    # trans = fitz.Matrix(1, 1).preRotate(0)
        pix = p.get_pixmap(matrix=fitz.Matrix(3, 3))
        # output = f'r{p.number}.png'
        pix.save(img_path + "/" + name + "_%d.jpg" % i)


files = os.listdir(img_path)
print(len(files))
files1 = os.listdir(pdfs_path)
print(len(files1))

        



