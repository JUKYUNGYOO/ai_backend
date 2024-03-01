
select_inference_ready_files = """
    SELECT seq_no, hist_pkey, file_seq_no, file_path
      FROM dbo.infer_ready_queue
     WHERE file_seq_no = :file_seq_no AND file_path LIKE '%.jpg%'
"""
select_inference_ready_files_all = """
    SELECT seq_no, hist_pkey, file_seq_no, file_path
      FROM dbo.infer_ready_queue
"""
