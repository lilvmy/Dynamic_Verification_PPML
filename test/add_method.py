import fitz
import os
from PIL import Image


# def compress_pdf_fitz(input_pdf, output_pdf, target_size_kb=2048, min_quality=30):
#     """使用 PyMuPDF + PIL 动态调整压缩 PDF，确保清晰度并小于目标大小"""
#
#     scale_factor = 1.5  # 初始缩放比例
#     image_quality = 85  # 初始 JPEG 质量
#     temp_img_path = "temp_img.jpg"  # 临时存储优化后的图片
#
#     while True:
#         doc = fitz.open(input_pdf)
#         new_doc = fitz.open()
#
#         for page in doc:
#             pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))  # 适当缩放，保持清晰度
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # 转换为 PIL 图像
#             img.save(temp_img_path, "JPEG", quality=image_quality)  # 设置 JPEG 质量
#
#             rect = page.rect
#             img_page = new_doc.new_page(width=rect.width, height=rect.height)
#             img_page.insert_image(rect, filename=temp_img_path)  # 插入优化后的图像
#
#         new_doc.save(output_pdf, garbage=4, deflate=True)  # 清理垃圾，优化 PDF
#         new_doc.close()
#         doc.close()
#
#         file_size_kb = os.path.getsize(output_pdf) / 1024  # 计算 PDF 大小
#         print(f"当前质量: {image_quality}, 缩放: {scale_factor}, 生成大小: {file_size_kb:.2f} KB")
#
#         if file_size_kb <= target_size_kb:  # 满足 2MB 目标，停止
#             break
#         elif image_quality > min_quality:  # 先降低 JPEG 质量
#             image_quality -= 5
#         elif scale_factor > 1.0:  # 若质量已到最低，则降低缩放比例
#             scale_factor -= 0.1
#         else:
#             print("⚠️ 无法进一步压缩，可能无法达到目标大小")
#             break  # 无法再缩小，则停止
#
#     os.remove(temp_img_path)  # 删除临时图片
#     print(f"✅ 最终压缩完成: {output_pdf}，最终大小: {file_size_kb:.2f} KB")
#
#
# # 示例调用
# compress_pdf_fitz("/home/lilvmy/Distributed_Optimization_in_Networked_Systems-Algorithms_and_Applications.pdf",
#                   "/home/lilvmy/compressed_output.pdf", target_size_kb=2048)

from Pyfhel import Pyfhel, PyPtxt, PyCtxt

HE = Pyfhel()
HE.contextGen(scheme='bfv', n=2**12, t_bits=20)
HE.keyGen()
HE.relinKeyGen()

ctxt1 = HE.encrypt(42)
ptxt1 = HE.encode(0)

r = ctxt1*ptxt1
print(r)