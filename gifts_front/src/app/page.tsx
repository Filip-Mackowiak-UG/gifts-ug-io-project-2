"use client";
import { SelectForm } from "@/app/components/SelectForm";
import { HomeHeader } from "@/app/components/HomeHeader";
import { Separator } from "@/components/ui/separator";
import { ProductCard } from "@/app/components/ProductCard";
import React, { useState } from "react";
import { Product } from "@/app/models/Product";
import { baseUrl } from "@/app/util/Constants";

export default function Home() {
  const [product, setProduct] = useState<Product>();
  const [loading, setLoading] = useState<boolean>(false);

  async function onClick(data: any) {
    setLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const product = await fetch(`${baseUrl}/get_recs`, {
      method: "POST",
      body: JSON.stringify(data),
      headers: {
        "Content-Type": "application/json",
      },
    });

    const productData: Product = await product.json();
    console.log("productData");
    setLoading(false);
    setProduct(productData);
  }

  return (
    <>
      <HomeHeader />
      <Separator className={`mb-4`} />
      <SelectForm onClick={onClick} />
      <ProductCard product={product} loading={loading} />
    </>
  );
}
