from KS.job.jobs import Jobs
from boostrap import Bootstrap

Bootstrap('ks')

jobs = Jobs()
jobs.create_jobs()
jobs.update_index()
jobs.run()



