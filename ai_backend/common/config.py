# db관련 config 클래스 
class Config:
    #####[SESSION]##### 
    SESSION_USE = True
    #여기에 설정된 URL 체크
    SESSION_CHECK_URLS = []
    SESSIONT_TIME = 60
    
    #####[DB:SQLAlchemy]##### DB 연결 정보임.
    # real / dev
    DB_ENV = 'dev'
    # SQLALCHEMY_SODDB_URI = 'mssql+pymssql://interminds:ntflow@192.168.0.85:11433/cspace_test'  #oriondb
    SQLALCHEMY_SODDB_URI = 'mssql+pymssql://soduser:orion!20231212@10.16.1.224:1433/soddb'  #oriondb
    # SQLALCHEMY_COMMON_DB_URI = 'mssql+pymssql://soduser:orion!20231212@10.16.1.224:1433'
    
    #####[SLACK]#####
    TOKEN = 'xoxb-1243170538740-3810385899846-HcA610BUZOlRDwgnzHSS5I8p'
    CHANNEL = 'C03QQLR3AM6'
    
    #[AWS]
    AWS_REGION = 'ap-northeast-2'
    AWS_USER_POOL_ID = 'ap-northeast-2_PwDOMlti8'
    AWS_APP_CLIENT_ID = '2vhbfq991eg4tde54drqfks69n'
    
    # db 설정. 