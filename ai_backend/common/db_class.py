# sod 관련 db
# config.py 설정 값으로 db연결과 관련된 객체 생성 및 관리.

from sql.delete_inference_queue_item import delete_inference_queue_item
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from common.util import *
from common.config import Config
from common.error_code.server_error_code import ServerErr
from sql.inference_ready_sql import select_inference_ready_files
from sql.insert_inference_results import insert_inference_results

# db engine
engine = create_engine(Config.SQLALCHEMY_SODDB_URI, poolclass=NullPool)
Session = scoped_session(sessionmaker(bind=engine))
# engine,session
engine_dev = create_engine(Config.SQLALCHEMY_SODDB_URI, poolclass=NullPool)
Session_dev = scoped_session(sessionmaker(bind=engine_dev))



# db공통기능
class DbClass():
    def __init__(self, logger, err_obj=None):
        self.logger = logger  
        self.session = Session_dev() if Config.DB_ENV == 'dev' else Session()
        self.err_obj = ServerErr() if err_obj is None else err_obj

    # delete
    def __del__(self):
        self.logger.debug('====================[SESSION CLOSE]====================')
        self.session.close()
    # commit
    def commit(self):
        self.logger.debug('====================[COMMIT]====================')
        self.session.commit()
    
    
    def rollback(self):
        self.logger.warning('====================[ROLLBACK]====================')
        self.session.rollback()

    # select one 
    def select_(self, sql, _dict={}):
        try:
            self.logger.info(f'[select_]\n{sql}') 
            res = self.session.execute(sql, _dict).first() 
            if res is not None:
                res = snake_to_camel_dict(res)  # snake to camel -> json때문에
        except Exception as err:
            self.err_obj.DB('1204')  # db 오류처리 
            raise Exception(f'[select_] 오류', err)
        else:
            self.logger.info(f'[select_] 결과 :\n {res}')  
            return res


    # DELETE 
    def delete_(self, sql, _dict={}):
        try:
            self.logger.info(f'[delete_]\n{sql}')
            self.session.execute(sql, _dict) 
        except Exception as err:
            self.err_obj.DB('1207')  
            raise Exception(f'[delete_] 오류', err)
        else:
            self.logger.info(f'[delete_] 성공!')  

    # INSERT 쿼리를 실행하는 메소드입니다.
    def insert_(self, sql, _dict={}):
        try:
            self.logger.info(f'[insert_]\n{sql}')  
            res = self.session.execute(sql, _dict)  # insert query
        except Exception as err:
            self.err_obj.DB('1205')  # error 
            raise Exception(f'[insert_] 오류', err)
        else:
            self.logger.info(f'[insert_] 성공!') 
            return res
        
    
    def insert_ret_(self, sql, _dict={}):
        try:
            self.logger.info(f'[insert_ret_]\n{sql}') 
            res = self.session.execute(sql, _dict).first()[0]  
        except Exception as err:
            self.err_obj.DB('1205')  
            raise Exception(f'[insert_ret_] 오류', err)
        else:
            self.logger.info(f'[insert_ret_] 성공!')  
            return res

    # UPDATE 쿼리를 실행하는 메소드입니다.
    def update_(self, sql, _dict={}):
        try:
            self.logger.info(f'[update_]\n{sql}')  
            self.session.execute(sql, _dict)  
        except Exception as err:
            self.err_obj.DB('1206')  
            raise Exception(f'[update_] 오류', err)
        else:
            self.logger.info(f'[update_] 성공!')  


    def update_ret(self, sql, _dict={}):
        try:
            self.logger.info(f'[update_ret]\n{sql}')  
            res = self.session.execute(sql, _dict) 
        except Exception as err:
            self.err_obj.DB('1206')  
            raise Exception(f'[update_ret] 오류', err)
        else:
            self.logger.info(f'[update_ret] 성공!') 
            return res
        
    
    def procedure_call(self, sql, _dict={}):
        try:
            self.logger.info(f'[procedure call]\n{sql}')  
            res = self.session.execute(sql, _dict).one()  
        except Exception as err:
            self.err_obj.DB('1204')  
            raise Exception(f'[procedure call] 오류', err)
        else:
            return res  
            # 호출 결과 반환.

    def delete_inference_queue_item_all(self, file_seq_no):
        try:
            self.logger.info('[delete_specific_item] Deleting item...')
            self.session.execute(delete_inference_queue_item, {'file_seq_no': file_seq_no})
            self.session.commit()
        except Exception as e:
            self.logger.error('Error deleting item', exc_info=True)
            self.session.rollback()
            raise
        else:
            self.logger.info('Item successfully deleted.')

            
    def insert_inference_results_method(self, data):
        try:
            self.logger.info(f'Inserting inference results: {data}')
        
        # # film_dt 값을 YYYYMMDD 형식의 문자열에서 datetime.date 객체로 변환
        #     if 'film_dt' in data and data['film_dt']:
        #         data['film_dt'] = datetime.strptime(data['film_dt'], '%Y%m%d').date()

        # Execute the SQL query with named parameters
            self.session.execute(insert_inference_results, data)
            self.session.commit()
            self.logger.info('Insertion successful.')
        except Exception as e:
            self.logger.error(f'Failed to insert inference results: {e}')
            self.session.rollback()
 
    def select_inference_ready_files(self, file_seq_no=None):
   
        params = {}
        if file_seq_no is not None:
            base_query += " AND file_seq_no = :file_seq_no"
            params["file_seq_no"] = file_seq_no

        try:
            self.logger.info('[select_inference_ready_files] Fetching .jpg files...')
            results = self.session.execute(base_query, params).fetchall()
            results = [dict(row) for row in results]
            if results:
                self.logger.info(f'[select_inference_ready_files] Success! Retrieved {len(results)} records.')
            else:
                self.logger.info('[select_inference_ready_files] No .jpg files found.')
            return results
        except Exception as e:
            self.logger.error('[select_inference_ready_files] Error fetching .jpg files', exc_info=True)
            raise

    def select_all_(self, sql, _dict=None):
        _dict = _dict or {} 
        try:
            self.logger.info(f'[select_all_]\n{sql}') 
            res = self.session.execute(sql, _dict).all()  
            result = [dict(row) for row in res]  # query to dict()
            result = snake_to_camel_arr(result)  #
        except Exception as err:
            self.err_obj.DB('1204')  # db error 처리
            raise Exception(f'[select_all_] 오류', err)
        else:
            self.logger.info(f'[select_all_] 성공!\t총 : {len(result)} ')  
            return result    
    def select_all_delete_infer_ready_queue(self, sql, _dict=None):
        try:
            self.logger.info(f'Selecting with SQL: {sql}')
            result = self.session.execute(sql, _dict if _dict else {}).fetchall()
            return [dict(row) for row in result]
        except Exception as e:
            self.logger.error('Error during select operation', exc_info=True)
            raise
    def delete_inference_queue_item_by_seq_no(self, file_seq_no):
        try:
            self.logger.info(f'[delete_inference_queue_item_by_seq_no] Deleting item with file_seq_no={file_seq_no} from inference queue...')
            # Execute the delete operation
            self.session.execute(delete_inference_queue_item, {'file_seq_no': file_seq_no})
            # Commit the changes to the database
            self.session.commit()
            self.logger.info(f'Successfully deleted item with file_seq_no={file_seq_no} from the inference queue.')
        except Exception as e:
            self.logger.error('Error while deleting item from the inference queue', exc_info=True)
            # Rollback in case of error
            self.session.rollback()
            raise
    def fetch_all_inference_ready_items(self):
        sql = """
            SELECT seq_no, hist_pkey, file_seq_no, file_path
            FROM dbo.infer_ready_queue
        """
        try:
            self.logger.info('[fetch_all_inference_ready_items] Fetching all items from inference queue...')
            result = self.session.execute(sql).fetchall()
            items = [dict(row) for row in result]  # 결과를 딕셔너리 리스트로 변환
            self.logger.info(f'[fetch_all_inference_ready_items] Successfully fetched {len(items)} items.')
            return items
        except Exception as e:
            self.logger.error('[fetch_all_inference_ready_items] Error fetching items from the inference queue', exc_info=True)
            self.session.rollback()
            raise

