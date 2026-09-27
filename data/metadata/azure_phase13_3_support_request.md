# CyberCodeMini — Phase 13.3: Azure for Students GPU Quota Support Request

## Executive Summary

- **Status**: `SUPPORT_REQUEST_DRAFT_READY`
- **Subscription Name**: `Azure for Students`
- **Subscription ID**: `4d6ad347-4459-4206-ac7c-54cc59637a4a`
- **Subscription Offer**: `MS-AZR-0170P`
- **RBAC Role**: `Owner`
- **Target Region**: `southafricanorth`
- **Target GPU**: `Standard_NV36ads_A10_v5` (1 × NVIDIA A10 24 GB)
- **Quota Family**: `StandardNVADSA10v5Family`
- **Required Quota**: `36 vCPUs`
- **Current Quota**: `0 vCPUs`
- **Azure Credit Spent**: `$0.00`
- **Subscription Changed**: `NO` (Subscription unchanged)

---

## Support Request Specification

| Parameter | Field Value |
| :--- | :--- |
| **Problem Type** | `Service and subscription limits (quotas)` |
| **Service** | `Azure Machine Learning` |
| **Region** | `South Africa North` |
| **Quota** | `Standard NVADSA10v5 Family Cluster Dedicated vCPUs` |
| **Requested Limit** | `36` (vCPUs) |
| **Current Limit** | `0` (vCPUs) |

---

## Factual Request Description Text

```text
I am using an Azure for Students subscription for an educational machine-learning project called CyberCodeMini. I am requesting a quota exception for one Azure Machine Learning GPU node in South Africa North.

The requested VM size is Standard_NV36ads_A10_v5, requiring 36 vCPUs from the StandardNVADSA10v5Family quota.

The current quota is 0 vCPUs. The Azure CLI quota API returns ResourceNotAvailableForOffer when attempting to request an increase, and the Azure Portal does not provide an adjustable quota for this Azure for Students subscription.

My account has Owner RBAC permissions on the subscription, so this does not appear to be an RBAC permission issue.

I am requesting whether Microsoft can grant a manual exception or otherwise enable 36 vCPUs for this GPU family for the Azure for Students subscription.

This request is for a single controlled educational training experiment. No GPU compute resources or training jobs have been created yet, and no Azure credit has been consumed.

If this quota cannot be granted under the Azure for Students offer, please confirm that explicitly and advise whether there is an eligible Azure subscription option that would allow the educational workload.
```

---

## Factual Representation Compliance

- Commercial production deployment claimed: **NO**
- Enterprise workload claimed: **NO**
- Guaranteed future Azure spending claimed: **NO**
- Multi-node / multi-GPU training claimed: **NO**
- Single controlled educational experiment requested: **YES (1 node / 36 vCPUs)**

---

## Next Steps for User

To submit this support request:
1. Navigate to [Azure Portal Help & Support](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade/newsupportrequest).
2. Create a new support ticket with Issue Type: **Service and subscription limits (quotas)**.
3. Select Quota type: **Azure Machine Learning**.
4. Set Region: **South Africa North** and Quota: **Standard NVADSA10v5 Family Cluster Dedicated vCPUs**.
5. Set New Limit: **36**.
6. Paste the exact factual request description text provided above.
7. Once submitted, record the Support Case Number to update this status to `SUPPORT_REQUEST_SUBMITTED`.

---

## Cost Protection Audit

All cost preservation constraints remain enforced:
- **Subscription Upgrade**: `NOT EXECUTED` (Preserved as Azure for Students)
- **Pay-As-You-Go Billing**: `NOT CREATED`
- **Azure GPU Compute Clusters**: `0`
- **Virtual Machines Created**: `0`
- **Azure ML Training Jobs**: `0`
- **Azure Credit Spent**: `$0.00`
- **Frozen Training Dataset**: `v0.3.0` (SHA-256: `c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5`)
