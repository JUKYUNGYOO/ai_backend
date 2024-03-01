from .common_error_code import CommonErr
from .error_code_define import *

class ServerErr(CommonErr):
    def __init__(self, ):
        super().__init__()
        self.err_type = 'Server Error Code'
        self.top_err_kind = '1'
        
    def Etc(self, err_code):
        self.resultCode = err_code              # 원본 에러 코드 (xxxx)
        self.mid_err_kind = '5'                 # [중분류] 코드
        self.code_dict = etc_error              # [소분류] 코드 정의 값 (error_code_define.py)
        self.check_err_code()                   # 코드 점검 및 코드번호 및 메시지 정의.
        
    def Etc_web(self, err_code):
        self.err_type = 'Web Error Code'
        self.resultCode = err_code              # 원본 에러 코드 (xxxx)
        self.mid_err_kind = '5'                 # [중분류] 코드
        self.code_dict = web_etc_error          # [소분류] 코드 정의 값 (error_code_define.py)
        self.check_err_code()                   # 코드 점검 및 코드번호 및 메시지 정의.
    
