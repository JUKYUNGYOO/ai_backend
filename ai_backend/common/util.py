import re  
import docker  
import json 
import subprocess  
from datetime import datetime  
from glob import glob  # 파일경로 검색
import pandas as pd  # 데이터 분석을 위한 모듈

# snake_case 문자열을 camelCase로 변환하는 함수
def snake_to_camel_dict(_old_dict):
    new_dict = {}
    old_dict_keys = _old_dict.keys()
    for old_key in old_dict_keys:
        new_dict[f"{snake_to_camel_str(old_key)}"] = _old_dict[old_key]
    return new_dict

# 리스트 내의 모든 사전을 snake_case에서 camelCase로 변환하는 함수
def snake_to_camel_arr(_old_arr):
    new_arr = []
    for _old_dict in _old_arr:
        new_dict = snake_to_camel_dict(_old_dict)
        new_arr.append(new_dict)
    return new_arr

# snake_case 문자열을 camelCase 문자열로 변환하는 함수
# def snake_to_camel_str(_snake):
#     titleStr =  _snake.title().replace("_", "")
#     camelStr = titleStr[0].lower() + titleStr[1:]
#     return camelStr
def snake_to_camel_str(_snake):
    if not _snake:
        return ""
    
    titleStr = _snake.title().replace("_", "")
    camelStr = titleStr[0].lower() + titleStr[1:]
    return camelStr


# camelCase 문자열을 snake_case 문자열로 변환하는 함수
def camel_to_snake_str(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def convert_phone_format(_phone_str=''):
    #지역번호로 판단하기
    _first = ''
    _middle = ''
    _last = ''
    if _phone_str.startswith("02"):
        _first = _phone_str[:2]
        if len(_phone_str[:2]) == 8:
            _middle = _phone_str[2:6]
            _last = _phone_str[6:10]
        else:
            _middle = _phone_str[2:5]
            _last = _phone_str[5:10]
    else:
        _first = _phone_str[:3]
        if len(_phone_str[3:]) == 8:
            _middle = _phone_str[3:7]
            _last = _phone_str[7:11]
        else:
            _middle = _phone_str[3:6]
            _last = _phone_str[6:10]

    return f'{_first}-{_middle}-{_last}'

# MS SQL 결과 리스트의 인코딩을 변환하는 함수
def ms_sql_incoding(ms_result_list):
    if ms_result_list:
        if isinstance(ms_result_list, list):
            for ms_result in ms_result_list:
                for key, value in ms_result.items():
                    try: # 문자열이 아닌 경우
                        # ms_result[key] = value.encode('ISO-8859-1').decode('cp949')
                        ms_result[key] = value.encode('ISO-8859-1').decode('utf-8')
                    except Exception as E:
                        continue
        else:
            for key, value in ms_result_list.items():
                    try:
                        # ms_result_list[key] = value.encode('ISO-8859-1').decode('cp949')
                        ms_result_list[key] = value.encode('ISO-8859-1').decode('utf-8')
                    except Exception as E:
                        continue

    return ms_result_list

# 학습 로그 url 함수에서 현재 학습 컨테이너가 존재하는지 체크하는 함수
def check_container_status(container_name):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        print(f'{container_name} : container.status : {container.status}')
        #logger.debug(f"{container_name} : container.status : {container.status}")
        # running or exited
        return True
    except docker.errors.NotFound:
        return False

# python 3.5 버전 이상에서 사용가능
# 학습 시작 버튼 클릭 시 docker 컨테이너를 생성, 실행하는 함수
def run_docker_container():
    try:

        today_date_time = datetime.now().strftime('%Y%m%d')[2:]
        excel_path = glob(f'/data01/dataset/base/annotations/*.xlsx')[0]
        num_class = str(len(pd.read_excel(f'{excel_path}', index_col = 0 )))

        set_training_start_json("train_model_folder_path", f"/data01/backend/output/{today_date_time}_{num_class}")

        # docker_command = ['docker', 'run', '-d', '--rm', \
        #                 '--name', 'orion-train', \
        #                 '--gpus', 'all', '--ipc=host', \
        #                 '-v', '/data01/backend:/workspace',\
        #                 '-v', '/data01/dataset:/workspace/dataset', \
        #                 '-v' , '/data01/frontend/training_start_time.json:/workspace/training_start_time.json', \
        #                 'docker.repository/user/orion:2.1.0-dev', \
        #                 'python', 'train.py', \
        #                 '--date', today_date_time, \
        #                 '--num_class', num_class]

        result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # docker container를 실행하는 명령어
        if result.returncode == 0:
            container_id = result.stdout.strip()
            return f"Container {container_id} started successfully!"
        else:
            error_message = result.stderr.strip()
            return f"Error starting container: {error_message}"
    except subprocess.CalledProcessError as e:
        return f"Error running docker command: {str(e)}"
    
# training_start_json 파일을 수정하는 함수(현재 학습 중 여부를 판단하는 파일을 수정)
def set_training_start_json(json_key, json_value):
    pass
    # json 파일로 덮어쓰고 읽기
    # with open(config['filepath']['training_start_json_path'], 'r') as json_file:
    #     json_data = json.load(json_file)
    # json_data[json_key] = json_value

    # with open(config['filepath']['training_start_json_path'], 'w', encoding='utf-8') as json_file:
    #     json.dump(json_data, json_file, indent="\t")