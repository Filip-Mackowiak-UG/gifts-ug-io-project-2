import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Product } from "@/app/models/Product";

export function ProductCard({
  product,
  loading,
}: {
  product: Product | undefined;
  loading: boolean;
}) {
  if (product === undefined && !loading) {
    return null;
  }
  if (loading) {
    return (
      <div className={`pt-4 flex flex-col justify-center items-center`}>
        <Card className={`w-1/4 p-4`}>
          <CardTitle>Loading...</CardTitle>
        </Card>
      </div>
    );
  }
  return (
    <div className={`pt-4 flex flex-col justify-center items-center`}>
      <Card className={`w-1/4 p-4`}>
        <CardTitle>{product?.name}</CardTitle>
        <CardDescription className={`mt-2`}>
          {product?.description}
        </CardDescription>
        <CardDescription className={`mt-2`}>
          Price: {product?.price}
        </CardDescription>
        <CardDescription className={`mt-2`}>
          Probability: {product?.probability.toFixed(6)}
        </CardDescription>
        <CardDescription className={`mt-2`}>
          <a href={product?.url} target={`_blank`}>
            Product Link
          </a>
        </CardDescription>
      </Card>
    </div>
  );
}
