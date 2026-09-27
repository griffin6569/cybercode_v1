# CyberCodeMini Azure Phase 13 Preflight & Quota Audit Report

**Date**: 2026-09-27  
**Phase**: Phase 13 Preflight  
**Status**: **`QUOTA_REQUEST_REQUIRED`**  

---

## 1. Subscription & Workspace Information

- **Subscription Name**: `Azure for Students`
- **Subscription ID**: `4d6ad347-4459-4206-ac7c-54cc59637a4a`
- **Tenant ID**: `20a12a57-3641-40d0-824b-f0e42f308ce6`
- **Resource Group**: `cybercodemini-rg` (`eastus`)
- **Azure ML Workspace**: `cybercodemini-ml` (`southafricanorth`)
- **Workspace Discovery URL**: `https://southafricanorth.api.azureml.ms/discovery`

---

## 2. Region Policy & Provider Registrations

- **Allowed Subscription Regions**: `indiasouthcentral`, `southafricanorth`, `italynorth`, `austriaeast`, `uaenorth`
- **Provider Registration Status**:
  - `Microsoft.Compute`: **Registered**
  - `Microsoft.MachineLearningServices`: **Registered**
  - `Microsoft.Quota`: **Registered**

---

## 3. GPU SKU Availability & Quota Analysis

- **Target GPU SKU**: `Standard_NV36ads_A10_v5`
- **GPU Specs**: 1 $\times$ NVIDIA A10 GPU (24 GB VRAM), 36 vCPUs, 110 GB RAM
- **Azure ML Quota Family Name**: `StandardNVADSA10v5Family` ("Standard NVADSA10v5 Family Cluster Dedicated vCPUs")
- **Current Quota Limit**: **`0`** vCPUs (`isQuotaApplicable: true`)
- **Existing Quota Requests**: `[]` (0 pending, 0 approved, 0 rejected)

---

## 4. Minimum Quota Required & Requestability

- **Required Quota Value**: **36 vCPUs** in `StandardNVADSA10v5Family` for location `southafricanorth`.
- **Explanation**: Azure ML measures quota in dedicated vCPUs per VM family. One instance of `Standard_NV36ads_A10_v5` contains 36 vCPUs.
- **Requestability Audit**:
  - `Azure for Students` subscriptions default GPU quota limits to 0.
  - Quota increase can be requested via Azure Portal under [My Quotas](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas) or by submitting a support request ("Service and subscription limits").

---

## 5. Cost Protection & Verified Pricing

1. **Hourly Compute Rate**: ~$1.85 / hour for `Standard_NV36ads_A10_v5` in `South Africa North`.
2. **Student Credit Applicability**: The $100 student credit covers Pay-As-You-Go compute costs.
3. **Azure ML Overhead**: $0.00 (No extra management fee for `AmlCompute`).
4. **Non-Compute Storage/Logging Charges**: Currently < $0.05 / month in `cybercodemini-rg`.
5. **Estimated Single Pilot Job Cost**:
   - Runtime: ~0.35 hours (20 mins) $\times$ $1.85/hr = **~$0.65 – $0.70 total cost**.
   - Preserves **> $99.00** of the $100 student credit.

---

## 6. Planned Compute & Pilot Job Design

### Planned AmlCompute Config (To deploy after quota approval):
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json
name: cybercode-a10-compute
type: amlcompute
size: Standard_NV36ads_A10_v5
location: southafricanorth
min_instances: 0
max_instances: 1
idle_time_before_scale_down: 120
tier: Dedicated
```

### Planned Pilot Job Config:
- **Corpus Version**: Frozen `v0.3.0` (`training.jsonl` SHA-256: `c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5`)
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Method**: QLoRA 4-bit NF4 double quantization
- **LoRA Hyperparameters**: $r=16, \alpha=32, \text{dropout}=0.05$ across 7 target modules
- **Instance Safety**: `max_instances=1`, `max_retries=0`, `min_instances=0`

---

## 7. Safety Verification & Current Resource Status

- **Azure GPU Compute Instances Created**: **0**
- **Azure ML Training Jobs Submitted**: **0**
- **Azure Credit Spent**: **$0.00**

---

## 8. Exact Next Action for User

To proceed to the Azure GPU pilot:
1. Open Azure Portal: [https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas)
2. Select Provider: **Azure Machine Learning**
3. Select Region: **South Africa North**
4. Locate Quota Family: **Standard NVADSA10v5 Family Cluster Dedicated vCPUs**
5. Request Quota Increase to: **36 vCPUs** (or 1 node).

---

## 9. Final Status Decision

**FINAL STATUS**: **`QUOTA_REQUEST_REQUIRED`**
