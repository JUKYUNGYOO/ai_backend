# delete_inference_queue_item = """
#     DELETE 
#       FROM dbo.infer_ready_queue 
#      WHERE seq_no = :seq_no
# """

delete_inference_queue_item = """
    DELETE 
      FROM dbo.infer_ready_queue 
     WHERE file_seq_no = :file_seq_no
"""
delete_inference_item_all = """
     DELETE FROM dbo.infer_ready_queue
"""