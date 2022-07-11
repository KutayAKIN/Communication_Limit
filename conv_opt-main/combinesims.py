from utils import combine_sims

#run_ids = [i for i in range(204,334)]
run_ids = [i for i in range(3,23)]

sim_type = "BDD"
device_no = 2

run_loc = "./runs/"+sim_type+"/device"+str(device_no)

target_loc = "./combined/"+sim_type

combine_sims(run_ids,run_loc,target_loc)