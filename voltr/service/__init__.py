"""The daily paper-trading service: one UDP receiver fanned out to isolated pipeline workers.

    supervisor.py  daily schedule, contract download, worker launch/restart, status
    worker.py      one pipeline per process: decodes its private feed copy and drives the adapter
    fanout.py      framed raw-datagram channels from the receiver to each worker
    redirect.py    keeps old dashboard bookmark ports pointing at the live dashboard
    README.md      how isolation, journaling and recovery work
"""
