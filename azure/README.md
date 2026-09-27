# CyberCodeMini Azure Setup Guide

## ⚠️ Cost Protection

This project is designed for Azure for Students with a limited credit budget.
**All GPU operations require explicit human confirmation.**

## Prerequisites

1. Azure for Students subscription
2. Azure CLI installed (`az`)
3. Python 3.11+

## Step-by-Step Setup

### 1. Login to Azure

```bash
az login
az account list --output table
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
```

### 2. Create Resource Group

```bash
az group create \
  --name cybercodemini-rg \
  --location eastus
```

### 3. Create Azure ML Workspace

```bash
az ml workspace create \
  --name cybercodemini-ws \
  --resource-group cybercodemini-rg \
  --location eastus
```

### 4. Select GPU Compute

Check available GPU VMs for your subscription:

```bash
az vm list-sizes --location eastus --output table | grep -i "nc\|nd"
```

Recommended for budget training:
- `Standard_NC6s_v3` (1x V100 16GB) — ~$3/hr
- `Standard_NC4as_T4_v3` (1x T4 16GB) — ~$0.53/hr ← **Best for students**

### 5. Create Compute Target (Manual Only)

```bash
az ml compute create \
  --name gpu-cluster \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws \
  --type AmlCompute \
  --size Standard_NC4as_T4_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 600
```

**Important:** Set `--min-instances 0` so compute scales to zero when idle.

### 6. Run Preflight Check

Before submitting any training job:

```bash
python azure/scripts/preflight.py
```

This will display:
- Model and dataset configuration
- Estimated training duration
- Estimated cost
- **Requires explicit confirmation before proceeding**

### 7. Submit Training Job

```bash
az ml job create --file azure/train_job.yaml \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws
```

### 8. Monitor Job

```bash
az ml job show --name <job_name> \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws
```

Or use the Azure ML Studio web UI.

### 9. Download Artifacts

```bash
az ml job download --name <job_name> \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws \
  --output-name outputs
```

### 10. Stop/Delete Compute

**Always verify compute is stopped after training:**

```bash
# Check compute status
az ml compute show --name gpu-cluster \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws

# Delete compute when done
az ml compute delete --name gpu-cluster \
  --resource-group cybercodemini-rg \
  --workspace-name cybercodemini-ws --yes
```

## Environment Variables

Set in `.env` (never committed):

```
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=cybercodemini-rg
AZURE_WORKSPACE=cybercodemini-ws
AZURE_COMPUTE_NAME=gpu-cluster
```

## Cost Monitoring

```bash
# Check current spending
az consumption usage list \
  --start-date 2026-09-01 \
  --end-date 2026-09-30 \
  --output table
```

## ⚠️ Never

- Never commit Azure credentials or API keys
- Never leave GPU compute running unattended
- Never set `min-instances > 0` on GPU compute
- Never skip the preflight check
