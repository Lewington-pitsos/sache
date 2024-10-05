module "sache_production" {
  source = "../../main"
  instance_type = "g6e.2xlarge"
}


# "g6e.4xlarge" # 20 Gbps bandwidth = 2500 MBps, 48 GPU ram, 16 VCPU
# 625 MBps on a g6e.2xlarge instance
# (8 layer * 50 * 257 * 1024) / 1024 / 1024 = 100 MBps requried @ 50 images per second

# 10000000 / 50 /60/60 = 55 hours @ 50 images per second