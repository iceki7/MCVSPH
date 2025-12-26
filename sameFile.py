
import hashlib
import os


#COPY
 
def calculate_file_md5(file_path):
    with open(file_path, 'rb') as file:
        file_content = file.read()  # 一次性读取文件内容
 
    md5_value = hashlib.md5(file_content).hexdigest()
    return md5_value


#hash一致说明文件确实是一致的，但如果hash不一致，也不一定文件内容不一致。
def delCopy(keepfile,delfile):
    
    x=calculate_file_md5(keepfile)
    y=calculate_file_md5(delfile)
    if(x==y):
        print('same')
    else:
        print('diff')



predir=r"D:\vscode-portable\portapp - zxctemp.json"
# //支持改名
# //支持修改内容，可以改过去再改回来

print('[dir]\t'+predir)
print('[md5]\t'+calculate_file_md5(predir))