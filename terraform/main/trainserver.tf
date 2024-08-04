resource "aws_instance" "sache_train" {
  ami           = "ami-0197c13a4f68c9360"
  instance_type = var.training_server_instance_type
  key_name      = aws_key_pair.deployer.key_name

  security_groups = [aws_security_group.allow_ssh.name]

  tags = {
    Name = "sache_train"
  }

  root_block_device {
    volume_size = 400
  }
}