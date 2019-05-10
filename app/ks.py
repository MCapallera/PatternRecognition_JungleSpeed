import logging
from KS.job.jobs import Jobs
from boostrap import Bootstrap


if __name__ == '__main__':
    Bootstrap('ks')
    jobs = Jobs()
    jobs.create_jobs()
    jobs.update_index()
    jobs.run()
else:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(name)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


