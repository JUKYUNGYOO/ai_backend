SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE';

SELECT * from sh_upload_hist_file;
SELECT * from infer_ready_queue;
SELECT * from infer_insert_test;
select * from SHELF_PRODUCT;
Drop table infer_ready_queue;
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'shelf_product';
ALTER TABLE SHELF_PRODUCT ADD sum_area BIGINT;

SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'SHELF_PRODUCT';

SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'infer_insert_test';


-- CREATE TABLE infer_ready_queue (
-- 	seq_no bigint IDENTITY(1,1) NOT NULL,
-- 	hist_pkey bigint NOT NULL,
-- 	file_seq_no int NOT NULL,
-- 	file_path varchar(255) COLLATE Korean_Wansung_CI_AS DEFAULT '' NOT NULL,
-- 	CONSTRAINT tr_ready_queue_pk PRIMARY KEY (seq_no,hist_pkey,file_seq_no)
-- );
-- /sod/upload/00100_00564433/20240101/00000.jpg
CREATE TABLE infer_ready_queue (
	seq_no bigint IDENTITY(1,1) NOT NULL,
	hist_pkey bigint NOT NULL,
	file_seq_no int NOT NULL,
	file_path varchar(255) COLLATE Korean_Wansung_CI_AS DEFAULT '' NOT NULL,
	CONSTRAINT tr_ready_queue_pk PRIMARY KEY (seq_no,hist_pkey,file_seq_no)
);

INSERT INTO infer_ready_queue (hist_pkey, file_seq_no, file_path) VALUES
(1, 1, '/sod/upload/00100_80004173/20240228/1_KakaoTalk_20240117_104750882.jpg'),
(2, 2, '/sod/upload/00100_80004173/20240228/2_KakaoTalk_20240117_104750882_01.jpg');

-- INSERT INTO infer_ready_queue (hist_pkey, file_seq_no, file_path) VALUES
-- (1, 1, '/sod/upload/00100_80004173/20240228/1_KakaoTalk_20240117_104750882.jpg'),
-- (2, 2, '/sod/upload/00100_80004173/20240228/1_KakaoTalk_20240117_104750882.jpg'),
-- (3, 3, '/sod/upload/00100_00564433/20240101/00002.jpg'),
-- (4, 4, '/sod/upload/00100_00564433/20240101/00006.jpg'),
-- (5, 5, '/sod/upload/00100_00564433/20240101/00010.jpg'),
-- (6, 6, '/sod/upload/00100_00564433/20240101/00011.jpg'),
-- (7, 7, '/sod/upload/00100_00564433/20240101/00012.jpg'),
-- (8, 8, '/sod/upload/00100_00564433/20240101/00013.jpg'),
-- (9, 9, '/sod/upload/00100_00564433/20240101/00014.jpg'),
-- (10, 10, '/sod/upload/00100_00564433/20240101/00015.jpg'),
-- (11, 11, '/sod/upload/00100_00564433/20240101/00016.jpg'),
-- (12, 12, '/sod/upload/00100_00564433/20240101/00018.jpg'),
-- (13, 13, '/sod/upload/00100_00564433/20240101/00019.jpg');

SELECT * from infer_ready_queue;

SELECT name
FROM sys.columns
WHERE object_id = OBJECT_ID('infer_ready_queue');

SELECT name, state_desc
FROM sys.databases;

SELECT seq_no, hist_pkey, file_seq_no, file_path FROM infer_ready_queue;


select * from SHELF_PRODUCT;

