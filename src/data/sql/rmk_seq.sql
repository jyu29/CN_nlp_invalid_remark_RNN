select seq_id, 
customer_remark,
"remark_Info"
from smartdata_pro.f_invalide_customer_remark
where seq_id >= {{seq_id}}