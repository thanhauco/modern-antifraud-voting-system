# Azure Government Terraform Configuration
# Modern Anti-Fraud Voting System Infrastructure

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.47.0"
    }
  }
  
  backend "azurerm" {
    resource_group_name  = "voting-tfstate-rg"
    storage_account_name = "votingtfstate"
    container_name       = "tfstate"
    key                  = "production.tfstate"
    environment          = "usgovernment"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
  environment = "usgovernment"
}

# Variables
variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  default     = "production"
}

variable "location" {
  description = "Azure Government region"
  type        = string
  default     = "usgovvirginia"
}

variable "aks_node_count" {
  description = "Number of AKS nodes"
  type        = number
  default     = 3
}

# Resource Group
resource "azurerm_resource_group" "voting" {
  name     = "voting-${var.environment}-rg"
  location = var.location
  
  tags = {
    Environment = var.environment
    Project     = "Modern Anti-Fraud Voting System"
    Compliance  = "FedRAMP-High"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "voting" {
  name                = "voting-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.voting.location
  resource_group_name = azurerm_resource_group.voting.name
}

resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.voting.name
  virtual_network_name = azurerm_virtual_network.voting.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_subnet" "db" {
  name                 = "db-subnet"
  resource_group_name  = azurerm_resource_group.voting.name
  virtual_network_name = azurerm_virtual_network.voting.name
  address_prefixes     = ["10.0.2.0/24"]
  
  service_endpoints = ["Microsoft.Sql"]
}

# Azure Kubernetes Service
resource "azurerm_kubernetes_cluster" "voting" {
  name                = "voting-${var.environment}-aks"
  location            = azurerm_resource_group.voting.location
  resource_group_name = azurerm_resource_group.voting.name
  dns_prefix          = "voting-${var.environment}"
  kubernetes_version  = "1.28"
  
  default_node_pool {
    name                = "system"
    node_count          = var.aks_node_count
    vm_size             = "Standard_D4s_v3"
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = 3
    max_count           = 10
    
    upgrade_settings {
      max_surge = "33%"
    }
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
  }
  
  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
    admin_group_object_ids = []
  }
  
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.voting.id
  }
  
  tags = {
    Environment = var.environment
    Compliance  = "FedRAMP-High"
  }
}

# Azure Database for PostgreSQL
resource "azurerm_postgresql_flexible_server" "voting" {
  name                   = "voting-${var.environment}-psql"
  resource_group_name    = azurerm_resource_group.voting.name
  location               = azurerm_resource_group.voting.location
  version                = "16"
  delegated_subnet_id    = azurerm_subnet.db.id
  administrator_login    = "votingadmin"
  administrator_password = var.db_password
  zone                   = "1"
  storage_mb             = 65536
  sku_name               = "GP_Standard_D4s_v3"
  
  high_availability {
    mode = "ZoneRedundant"
  }
  
  maintenance_window {
    day_of_week  = 0
    start_hour   = 2
    start_minute = 0
  }
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "voting" {
  name                = "voting-${var.environment}-redis"
  location            = azurerm_resource_group.voting.location
  resource_group_name = azurerm_resource_group.voting.name
  capacity            = 2
  family              = "P"
  sku_name            = "Premium"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  redis_configuration {
    enable_authentication = true
  }
}

# Key Vault
resource "azurerm_key_vault" "voting" {
  name                = "voting-${var.environment}-kv"
  location            = azurerm_resource_group.voting.location
  resource_group_name = azurerm_resource_group.voting.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"
  
  purge_protection_enabled   = true
  soft_delete_retention_days = 90
  
  enable_rbac_authorization = true
  
  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "voting" {
  name                = "voting-${var.environment}-logs"
  location            = azurerm_resource_group.voting.location
  resource_group_name = azurerm_resource_group.voting.name
  sku                 = "PerGB2018"
  retention_in_days   = 90
}

# Data sources
data "azurerm_client_config" "current" {}

# Outputs
output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.voting.name
}

output "aks_cluster_fqdn" {
  value = azurerm_kubernetes_cluster.voting.fqdn
}

output "postgresql_server_fqdn" {
  value = azurerm_postgresql_flexible_server.voting.fqdn
}

output "redis_hostname" {
  value = azurerm_redis_cache.voting.hostname
}
