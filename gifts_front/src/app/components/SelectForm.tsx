"use client";

import {zodResolver} from "@hookform/resolvers/zod";
import {useForm} from "react-hook-form";
import {z} from "zod";

import {Button} from "@/components/ui/button";
import {Form} from "@/components/ui/form";
import preferences from "../util/preferences.json";
import React from "react";
import DynamicSelect from "@/app/components/DynamicSelect";

const FormSchema = z.object({
    age: z.string().min(1, "Please select an age group."),
    gender: z.string().min(1, "Please select a gender."),
    hobby: z.string().min(1, "Please select a hobby."),
});

export function SelectForm({onClick}: { onClick?: (data: any) => void }) {

    const form = useForm<z.infer<typeof FormSchema>>({
        resolver: zodResolver(FormSchema),
    });

    function onSubmit(data: z.infer<typeof FormSchema>) {
        onClick?.(data);
    }

    return (
        <Form {...form}>
            <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="flex flex-col justify-center items-center"
            >
                <DynamicSelect
                    control={form.control}
                    name="age"
                    label="Age"
                    options={preferences.preferences.find((pref) => pref.category === "Age")?.options || []}
                    placeholder="Select your age group"
                />
                <DynamicSelect
                    control={form.control}
                    name="gender"
                    label="Gender"
                    options={preferences.preferences.find((pref) => pref.category === "Gender")?.options || []}
                    placeholder="Select your gender"
                />
                <DynamicSelect
                    control={form.control}
                    name="hobby"
                    label="Hobby"
                    options={preferences.preferences.find((pref) => pref.category === "Hobby")?.options || []}
                    placeholder="Select a hobby"
                />
                <Button className={`w-1/4 m-6`} type="submit">Submit</Button>
            </form>
        </Form>
    );
}
