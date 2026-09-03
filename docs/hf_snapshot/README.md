# Hugging Face Pro-day snapshot (2026-08-30)

Captured while the account still had `isPro: true`. Token redacted from every file
here. `whoami.json` also redacts email and the access-token object.

## Billing â€” is PAYG gated on Pro?

No. Hugging Face's Inference Providers pricing page
(https://huggingface.co/docs/inference-providers/en/pricing, fetched 2026-08-30)
says extra usage is pay-as-you-go for **Free, Pro, Team, and Enterprise**. Free
gets $0.10/month of inference credits; Pro gets $2.00/month of general-purpose
compute credits. After that, you buy more credits. Pro is not the switch that
turns PAYG on.

What Pro *does* buy that this snapshot is for: Hub rate limits, and the $2
included credit while it lasts.

`whoami-v2` on this account:

- `isPro`: true
- `canPay`: true
- `billingMode`: prepaid
- `periodEnd`: 1788220800 = **2026-08-31 00:00:00 UTC** (today is the last
  full calendar day in UTC+2)

Credit **balance** is not in the API. Paste it from
https://huggingface.co/settings/billing if you want the number next to this
file later. The code paths that spend it are synthesis, `ping`, and the
extraction runner â€” Hub retrieval does not.

## Catalog pin (extractor)

Hub card `openai/gpt-oss-120b`:

- revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
- `lastModified`: 2025-08-26T17:25:03.000Z

Router id on 2026-08-30 (136 models): `openai/gpt-oss-120b`. Live providers
included groq, novita, cerebras, nscale, together, fireworks-ai, scaleway.
`:fastest` is a routing alias, not a model. The extractor pin uses groq
(structured output, live) plus the Hub revision; the router call uses
`openai/gpt-oss-120b:groq` because the router does not take `@sha`.

See `extractor_pin.json`.

## Hub rate-limit headers (Pro)

From `GET .../Apertus-v1.5-8B/discussions?limit=5`:

```
RateLimit: "api";r=2487;t=155
RateLimit-Policy: "fixed window";"api";q=2500;w=300
```

That is a 2500-request / 300-second window with 2487 remaining at capture.
Compare after cutover; if the quota drops, this file is the before picture.

