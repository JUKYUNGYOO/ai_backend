from flask import Flask, request, jsonify
from flask_restx import Api, Resource, Namespace, fields
import logging
from werkzeug.datastructures import FileStorage
from PIL import Image, ImageOps
from io import BytesIO
import torch
import json 
from glob import glob
import os
import re
from common.db_class import DbClass
from common.error_code.server_error_code import ServerErr
from sql.inference_ready_sql import *
from orion import TwoStageOrionDetector, OrionDataManager
from flask_restx import reqparse
from datetime import datetime
from threading import Lock
from sql.insert_inference_results import insert_inference_results
from sql import *

is_inference_running = False
inference_lock = Lock()


def check_and_run_inference():
    global is_inference_running
    with inference_lock:
        if is_inference_running:
            return False
        is_inference_running = True
    return True

# inference 완료 시 is_inference_running = False로
def release_inference_lock():
    global is_inference_running
    logger.info("Releasing inference lock...")
    with inference_lock:
        is_inference_running = False
    logger.info("Inference lock released.")


parser = reqparse.RequestParser()
parser.add_argument('image', type=FileStorage, location='files', required=True, help='Image file to be processed.')
parser.add_argument('threshold', type=float, default=0.3, help='Detection threshold value.')
parser.add_argument('clf_threshold', type=float, default=0.7, help='Classification threshold value.')

app = Flask(__name__)
api = Api(app)
Inference = Namespace("inference", description="Inference operations")
api.add_namespace(Inference)
logger = logging.getLogger('admin_log')

# model_list = glob('./output/*_*') 상대경로
model_list = glob('/workspace/ai_backend/output/*_*') # 절대경로로

model_list.sort()
model_path = model_list[-1].split('/')[-1]
detector = TwoStageOrionDetector(
    # f'./output/{model_path}/test_model/model_final.pth',
    # f'./output/{model_path}/classifier/stage2/model_best.pth.tar',
    # glob(f'./output/{model_path}/*.xlsx')[-1],
    f'/workspace/ai_backend/output/{model_path}/test_model/model_final.pth',
    f'/workspace/ai_backend/output/{model_path}/classifier/stage2/model_best.pth.tar',
    glob(f'/workspace/ai_backend/output/{model_path}/*.xlsx')[-1],
    int(model_path.split('_')[-1])
)

@Inference.route('/dbconnect')
class DBConnectResource(Resource):
    def post(self):
        try:
            logger.info("Database connection successful.")
            return {"message": "db connect success"}, 200
        except Exception as e:
            logger.error("Database connection failed: ", exc_info=True)
            return {"message": "Database connection failed."}, 500

@Inference.route('/model-list')
class ModelListResource(Resource):
    def get(self):
        try:
            model_list = [i.split('/')[-1] for i in glob('./output/*_*')]
            return {"models": model_list}, 200
        except Exception as e:
            logger.error("Failed to retrieve model list: ", exc_info=True)
            return {"message": "Failed to retrieve model list."}, 500

def get_film_dt_and_codes_from_path(filePath):
    parts = filePath.split('/')
    try:
        # Assuming the filePath format is something like /path/to/acc_code/cus_code/film_dt/image.jpg
        # acc_code = parts[2]
        # print("acc_code ", acc_code)
        acc_code_cus_code = parts[-3]
        acc_code, cus_code = acc_code_cus_code.split('_',1)
        film_dt = str(parts[-2])
        # film_dt = parts[-2]
        return acc_code_cus_code,film_dt, acc_code, cus_code
    except IndexError:
        logger.error("Failed to extract film_dt, acc_code, and cus_code from filePath: {}".format(filePath))
        return None, None, None, None
#  해당 지점(acc_code, cus_code), 촬영날짜(film_dt) 에 대한 추론이 끝나면 infer_insert_test 테이블에 넣음.
def save_inference_result(result, original_file_path):
    # film_dt, acc_code_cus_code = get_film_dt_and_codes_from_path(original_file_path)
    acc_code_cus_code, film_dt, acc_code, cus_code = get_film_dt_and_codes_from_path(original_file_path) # 4개의 값을 올바르게 받도록 수정
    if film_dt is None or acc_code_cus_code is None:
        logger.error("One of film_dt, acc_code, or cus_code is None, cannot save inference result.")
        return
      # 'upload_' 접두사를 제거
    acc_code_cus_code = acc_code_cus_code.replace('upload_', '')

   
    # inference_result_path = f'/workspace/inference_result/{acc_code_cus_code}/{film_dt}/'
    inference_result_path = f'/workspace/inference_result/{acc_code_cus_code}/{film_dt}/'
     
    # if not os.path.exists(inference_result_path):
    #     os.makedirs(inference_result_path)
    if not os.path.exists(inference_result_path):
        os.makedirs(inference_result_path, exist_ok=True)
        # 디렉토리에 777 권한 설정
        os.chmod(inference_result_path, 0o777)
    
    original_file_name = os.path.basename(original_file_path)
    result_file_name = f"result_{original_file_name}.json"
    result_file_path = os.path.join(inference_result_path, result_file_name)

    try:
        with open(result_file_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
        logger.info("Inference result saved to {}".format(result_file_path))

        # DBClass인스턴스 생성.
        db_obj = DbClass(logger, ServerErr())
        process_json_files_in_directory(inference_result_path, db_obj)

#  inference_result_path = f'/workspace/inference_result/{acc_code_cus_code}/{film_dt}/'
    except Exception as e:
        logger.error(f"Failed to save inference result: {e}")

import hashlib

def generate_uid(index, json_file_name):
    # Create a unique string combining the index and the json file name
    unique_string = f"{index}_{json_file_name}"
    # Generate a hash of the unique string
    uid_hash = hashlib.sha256(unique_string.encode()).hexdigest()
    # Convert the hash to an integer
    uid_int = int(uid_hash, 16) % (10 ** 8)  # Reduce to a manageable integer size
    return uid_int

import os
path = "/workspace/inference_result"
os.chmod(path, 0o777)
# 추론 결과를 db에 저장.
def process_json_files_in_directory(directory_path, db):
    json_file = None
    try:
        parts = directory_path.strip('/').split('/')
        #inference_result/00100_00564433/20240101
        #  inference_result_path = f'/workspace/inference_result/{acc_code_cus_code}/{film_dt}/'

        # film_dt = parts[-2]
        # acc_code_cus_code = parts[-3]
        film_dt = parts[-1]
        acc_code_cus_code = parts[-2]
        try:
            acc_code, cus_code = acc_code_cus_code.split('_')
            print("film_dt, acc_code, cus_code", film_dt, acc_code, cus_code)
        except ValueError:
            logger.error(f"Unexpected format in {acc_code_cus_code}. Expected format: {{acc_code}}_{{cus_code}}")
            return  # 함수 종료

        print("film_dt:", film_dt)
        print("acc_code:", acc_code)
        print("cus_code:", cus_code)

        json_files = glob(os.path.join(directory_path, '*.json'))
        if not json_files:
            logger.info(f"No JSON files found in {directory_path}")
            return
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"{json_file} json files...")

                # Assuming additional data needed for insertion is extracted correctly
                # from either the JSON itself or defined elsewhere
                for index, product in enumerate(data.get('products', [])):
                    uid = index
                    # uid = generate_uid(index, os.path.basename(json_file))
                
                    product_name = product.get('name')
                    manufacturer = product.get('manufacturer')
                    # code = product.get('code')
                    counts = product.get('counts')
                    total_proportion = product.get('total_proportion')
                    # Assuming uid, category_nm, acc_code, cus_code, film_dt, sum_area are available
                    # uid = f"{os.path.basename(json_file).split('.')[0]}_{index}"
                    category_nm = "test-category-none"
                    acc_code = acc_code
                    cus_code = cus_code
                    film_dt = film_dt
                    sum_area = 3000

                    params = {
                        'uid': uid,
                        'product_name': product_name,
                        'manufacturers': manufacturer,
                        'product_count': counts,
                        'product_proportion': total_proportion,
                        'category_nm': category_nm,
                        'acc_code': acc_code,
                        'cus_code': cus_code,
                        'film_dt': film_dt,
                        'sum_area': sum_area
                    }
                     # 추가된 로그 출력 부분
                    logger.info(f"Inserting into database with params: {params}")
          
                    try:
                        db.insert_inference_results_method(params)
                        # db.insert_(insert_inference_results, params)
                        logger.info(f"Inserted product {product_name} to database successfully.")
                    except Exception as insert_error:
                        logger.error(f"Failed to insert product {product_name} into database: {insert_error}") 
    except Exception as e:
        if json_file:
            logger.error(f"Failed to process JSON file {json_file}: {e}")
        else:
            logger.error(f"Error processing JSON files in directory {directory_path}: {e}")


model_name = model_path
def perform_inference(filePath, image_content, threshold=0.3, clf_threshold=0.7):
    try:
        img = Image.open(BytesIO(image_content)).convert("RGB")
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logging.info({
            "status": "400",
            "type": "invalid_image_error",
            "exception": str(e),
        })
        return {"message": "Invalid image file. The file may be corrupted or not an image."}, 400, filePath

    try:
        with torch.no_grad():
            boxes, scores, preds = detector(img, threshold, clf_threshold)
        datamanager = OrionDataManager(boxes, scores, preds, detector.class_info, img, model_name)
        result = datamanager.result_extract()
        logging.info({"status": "200", "type": "inference_result"})
        return result, 200, filePath
    except Exception as e:
        logging.error({
            "status": "500",
            "type": "inference_error",
            "exception": str(e),
        })
        return {"message": "Model Inference Error"}, 500, filePath

 
def read_image_content(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()
@Inference.route("/inference-default")
class InferenceList(Resource):
    def post(self):
        if not check_and_run_inference():
            logger.error("Inference API has already been called.")
            return {"message": "Inference API has already been called."}, 429

        db_obj = DbClass(logger, ServerErr())
        try:
            logger.info("Fetching files for inference...")
            # infer_ready_rows = db_obj.select_all_(select_inference_ready_files_all)
            infer_ready_rows = db_obj.fetch_all_inference_ready_items()
            # file_seq_no = db_obj.select_all_delete_infer_ready_queue(select_inference_ready_files_all)
            if not infer_ready_rows:
                logger.info("No files found for processing.")
                return {"message": "No files found for processing."}, 404

            processed_files = 0

            for file_info in infer_ready_rows:
                filePath = file_info.get('file_path')
                file_seq_no = file_info.get('file_seq_no')  # file_seq_no 추출
                hist_pkey = file_info.get('hist_pkey')  # file_seq_no 추출
                print(f"Processing hist_pkey={hist_pkey} with filePath={filePath}")  # Printing file_seq_no and filePath
                
                print(f"Processing file_seq_no={file_seq_no} with filePath={filePath}")  # Printing file_seq_no and filePath
                logger.info(f"Processing file_seq_no={file_seq_no} with filePath={filePath}")  # Logging file_seq_no and filePath

                modified_path = filePath.replace('/sod/upload/', '/workspace/upload/')
                
                if not os.path.exists(modified_path):
                    logger.error(f"File not found: {modified_path}")
                    continue

                try:
                    image_content = read_image_content(modified_path)
                    result, status, _ = perform_inference(modified_path, image_content, threshold=0.3, clf_threshold=0.7)

                    if status == 200 and result:
                        # file_seq_no가 제공된 경우에만 삭제 작업 수행
                        if file_seq_no:
                            try:
                                db_obj.delete_inference_queue_item_by_seq_no(file_seq_no)  # 항목 삭제
                  
                                logger.info(f"Successfully deleted queue item for file_seq_no={file_seq_no}")
                                processed_files += 1
                            except Exception as delete_exception:
                                logger.error(f"Failed to delete queue item for file_seq_no={file_seq_no}: {delete_exception}")
                        else:
                            logger.warning(f"No file_seqNo provided for {modified_path}, cannot delete queue item.")
                        
                        save_inference_result(result, modified_path)
                        logger.info(f"Inference and result saving successful for {modified_path}")
                    else:
                        logger.error(f"Inference failed for {modified_path}, Status: {status}")
                except Exception as e:
                    logger.exception(f"Exception during processing {modified_path}: {e}")

            if processed_files > 0:
                return {"message": f"Processed {processed_files} files successfully."}, 200
            else:
                return {"message": "Failed to process any files due to errors."}, 500
        except Exception as e:
            logger.exception(f"Batch inference processing failed: {e}")
            return {"message": "Error processing the batch inference request."}, 500
        finally:
            release_inference_lock()
            logger.info("Inference lock released.")
            if db_obj:
                del db_obj


@Inference.route("/inference-combang-start")
class InferenceComList(Resource):
    @staticmethod
    def get():
        if not check_and_run_inference():
            return {"message": "Already inference API called"}, 429 
        db_obj = DbClass(logger, ServerErr())
        try:
            # jpg_files = db_obj.select_all_(select_inference_ready_files)
            
            # jpg_files = db_obj.select_all_(select_inference_ready_files, {"file_seq_no": file_seq_no})
            jpg_files = db_obj.fetch_all_inference_ready_items()
            # jpg_files = db_obj.select_all_(select_inference_ready_files, params)


            jpg_count = len(jpg_files)
            if jpg_count < 10:
                logger.info(f"Only {jpg_count} .jpg files found. At least 10 files required for batch processing.")
                return {"message": f"Only {jpg_count} .jpg files found. At least 10 files required for batch processing."}, 404

            processed_files = 0

            for file_info in jpg_files:
                filePath = file_info.get('filePath')
                # film_dt = get_film_dt_and_codes_from_path(filePath)
                # film_dt, acc_code, cus_code = get_film_dt_and_codes_from_path(filePath)
                acc_code_cus_code, film_dt, acc_code, cus_code = get_film_dt_and_codes_from_path(filePath)
                if not filePath:
                    logger.error(f"Missing 'filePath' in database result for record: {file_info}")
                    continue

                modified_path = filePath.replace('/sod/upload/', '/workspace/upload/')
                logger.info(f"Processing file: {modified_path}")
                try:
                    with open(modified_path, 'rb') as image_file:
                        image_content = image_file.read()
                        result, status, _ = perform_inference(filePath, image_content, threshold=0.3, clf_threshold=0.7)  # Correct parameter order
                        if status == 200:
                            file_seq_no = file_info.get('fileseqNo')
                            if file_seq_no is not None:
                                db_obj.delete_inference_queue_item(file_seq_no)
                                processed_files += 1
                           
                            
                            # seq_no = file_info.get('seqNo')
                            # if seq_no is not None:
                            #     db_obj.delete_inference_queue_item(seq_no)
                            #     processed_files += 1
                            
                            save_inference_result(result, modified_path)
                        else:
                            logger.error(f"Inference failed for file {modified_path}, Status: {status}")
                except IOError as e:
                    logger.error(f"Error opening or reading file {modified_path}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during inference for file {modified_path}: {e}")

            if processed_files > 0:
                return {"message": "Inference completed for batch."}, 200
            else:
                return {"message": "No valid files were processed due to errors."}, 500

        except Exception as e:
            logger.error(f"Error processing batch inference request: {e}", exc_info=True)
            return {"message": "Error processing request"}, 500
        finally:
            release_inference_lock()
            if db_obj is not None:
                del db_obj

                # dPrka!@01