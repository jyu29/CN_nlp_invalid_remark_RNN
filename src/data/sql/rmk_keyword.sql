select seq_id,
customer_remark,
"remark_Info" as 
remark_info from smartdata_pro.f_invalide_customer_remark 
where customer_remark 
like '%%AG'

