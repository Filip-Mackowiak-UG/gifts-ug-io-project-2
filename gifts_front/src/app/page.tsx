"use client";
import { SelectForm } from "@/app/components/SelectForm";
import { HomeHeader } from "@/app/components/HomeHeader";
import { Separator } from "@/components/ui/separator";
import { ProductCard } from "@/app/components/ProductCard";
import React, { useState } from "react";
import { Product } from "@/app/models/Product";
import { baseUrl } from "@/app/util/Constants";

export default function Home() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  async function onClick(data: any) {
    setLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const response = await fetch(`${baseUrl}/get_recs`, {
      method: "POST",
      body: JSON.stringify(data),
      headers: {
        "Content-Type": "application/json",
      },
    });

    const productData: Product[] = await response.json();
    console.log("productData", productData);

    // Sort products by probability in descending order
    productData.sort((a, b) => b.probability - a.probability);

    setLoading(false);
    setProducts(productData);
  }

  return (
    <>
      <HomeHeader />
      <Separator className={`mb-4`} />
      <SelectForm onClick={onClick} />
      {loading ? (
        <p className={`pt-4 flex flex-col justify-center items-center`}>
          Loading...
        </p>
      ) : (
        products.map((product) => (
          <ProductCard key={product.id} product={product} loading={loading} />
        ))
      )}
    </>
  );
}
