insert_inference_results = """
                    INSERT INTO soddb.dbo.SHELF_PRODUCT 
                            (uid, product_name, manufacturers, product_count, product_proportion, 
                            category_nm, acc_code, cus_code, film_dt, sum_area) 
                        VALUES 
                            (:uid, :product_name, :manufacturers, 
                                :product_count, :product_proportion, 
                            :category_nm, :acc_code, :cus_code, :film_dt, :sum_area)"""
