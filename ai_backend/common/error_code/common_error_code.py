from .error_code_define import *

# -------------------  [공통] 에러 클래스 (Parent)  -------------------
class CommonErr:
    def __init__(self, ):
        self.err_type = ''                                                  # Server / Raspberry Pi / Web / VD ...
        self.top_err_kind = '0'                                             # 1(Server) / 2(Raspberry) / 3(Web) / 4(VD) [대분류]
        self.mid_err_kind = '0'                                             # 1(S3) / 2(DB) / 3(Redis) / 4(Request) / 5(Etc) [중분류]
        self.resultCode = '000'                                             # Error Code (xy__) / default : '000'
        self.resultMsg = 'SUCCESS'                                          # Error Msg / default : 'SUCCESS'
        self.resultDict = {'resultCode' : '000', 'resultMsg' : 'SUCCESS'}   # Error Result ({resultCode : '', resultMsg : ''})
        self.code_dict = {}                                                 # Error Code Dict [error_code_define.py] [소분류 코드 정의]

    # Normal Code Set.
    def set_normal_code(self,):
        self.resultCode = '000'
        self.resultMsg = 'SUCCESS'
        self.resultDict = {'resultCode' : '000', 'resultMsg' : 'SUCCESS'}

    # Error Code Define Check.
    def check_err_code(self,):
        try:
            # xy__ => 대(x) / 중(y) / 소분류(__) 코드
            top_err_code = str(self.resultCode[0])
            mid_err_code = str(self.resultCode[1])
            bot_err_code = str(self.resultCode[2:])

            # 대/중분류 코드와 일치 여부 판단 / 소분류가 정의되어 있는지 판단.
            if top_err_code != self.top_err_kind:
                self.resultMsg = f'[{self.err_type}] Error_Code({self.resultCode}) 대분류 정의가 틀렸습니다.'
            elif mid_err_code != self.mid_err_kind:
                self.resultMsg = f'[{self.err_type}] Error_Code({self.resultCode}) 중분류 정의가 틀렸습니다.'
            elif bot_err_code not in self.code_dict.keys():
                self.resultMsg = f'[{self.err_type}] Error_Code({self.resultCode}) 소분류 코드값이 존재 하지 않습니다.'
            else:   # 정상 Case. / 코드 번호 및 에러메시지 정의.
                self.resultMsg = self.code_dict[self.resultCode[2:]]
        except Exception as e:
            self.resultMsg = f'[{self.err_type}] (check_err_code) Error({str(e)})'

        self.resultDict = {'resultCode': f'{self.resultCode}', 'resultMsg': f'{self.resultMsg}'}

    # [Common] DB Error Code Define.
    def DB(self, err_code):
        self.resultCode = err_code          # 원본 에러 코드 (xxxx)
        self.mid_err_kind = '2'             # [중분류] 코드
        self.code_dict = db_error          # [소분류] 코드 정의 값 (error_code_define.py)
        self.check_err_code()               # 코드 점검 및 코드번호 및 메시지 정의.

    
    # [Common] Request Error Code Define.
    def Request(self, err_code):
        self.resultCode = err_code          # 원본 에러 코드 (xxxx)
        self.mid_err_kind = '4'             # [중분류] 코드
        self.code_dict = request_error     # [소분류] 코드 정의 값 (error_code_define.py)
        self.check_err_code()               # 코드 점검 및 코드번호 및 메시지 정의.
    
    # [Common] Etc Error Code Define.
    def Etc(self, err_code):
        self.resultCode = err_code          # 원본 에러 코드 (xxxx)
        self.mid_err_kind = '5'             # [중분류] 코드
        self.code_dict = etc_error         # [소분류] 코드 정의 값 (error_code_define.py)
        self.check_err_code()               # 코드 점검 및 코드번호 및 메시지 정의.