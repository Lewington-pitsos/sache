module "sache_production" {
  source = "../../main"
  instance_type = "g4dn.8xlarge"
  training_server_instance_type = "g4dn.8xlarge"
}